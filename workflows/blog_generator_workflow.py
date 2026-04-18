import os
import sys
import re
from datetime import datetime
from typing import Optional
from langgraph.types import Send
from jinja2 import Template
from pathlib import Path
from typing import Optional, List


root_dir = Path(__file__).parent.parent

from logger import GLOBAL_LOGGER
from exception.custom_exception import BlogGeneratorException

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults



from utils.model_loader import ModelLoader
from prompt_library.prompt_locator import (DECIDE_IMAGES_SYSTEM_PROMPT, 
                                           ROUTER_SYSTEM_PROMPT,
                                           RESEARCH_SYSTEM_PROMPT,
                                           ORCH_SYSTEM_PROMPT,
                                           WORKER_SYSTEM_PROMPT)

from schemas.models import (State, 
                            GlobalImagePlan, 
                            Plan,
                            Task, 
                            RouterDecision, 
                            EvidenceItem, 
                            EvidencePack)


from workflows.image_workflow import MergeImagesWorker

class BlogGeneratorWorker:
    """
    Handles the end-to-end process of generating workflow using langgraph
    """
    
    def __init__(self,llm,image_llm):
        self.llm = llm
        self.image_llm = image_llm
        self.memory_saver = MemorySaver()
        self.logger = GLOBAL_LOGGER.bind(module="BlogGeneratorWorker")
        
        
    
    def _router_node(self,state: State) -> dict:
        
        try:
            self.logger.info("Planning blog")
            topic = state["topic"]
            decider = self.llm.with_structured_output(RouterDecision)
            decision = decider.invoke(
                [
                    SystemMessage(content=str(ROUTER_SYSTEM_PROMPT)),
                    HumanMessage(content=f"Topic: {topic}"),
                ]
            )
            self.logger.info("Planned blog successfully")
            
        except Exception as e:
            self.logger.error("Error planning blog", error=str(e))
            raise BlogGeneratorException("Error planning blog")

        return {
            "needs_research": decision.needs_research,
            "mode": decision.mode,
            "queries": decision.queries,
    }

    def _route_next(self,state: State) -> str:
        try:
            
            self.logger.info("Routing blog")
            deciding_node = "research" if state["needs_research"] else "orchestrator"
            self.logger.info("Routed blog successfully")
        except Exception as e:
            self.logger.error("Error routing blog", error=str(e))
            raise BlogGeneratorException("Error routing blog")
        
        return deciding_node
    
    
    # Tavily search
    def _tavily_search(self,query: str, max_results: int = 5) -> List[dict]:
        
        try:
            self.logger.info("Searching tavily")
            tool = TavilySearchResults(
                 tavily_api_key=os.getenv("TAVILY_API_KEY"),
                 max_results=max_results,)
            results = tool.invoke({"query": query})

            normalized: List[dict] = []
            for r in results or []:
                normalized.append(
                    {
                        "title": r.get("title") or "",
                        "url": r.get("url") or "",
                        "snippet": r.get("content") or r.get("snippet") or "",
                        "published_at": r.get("published_date") or r.get("published_at"),
                        "source": r.get("source"),
                    }
                )
                
            self.logger.info("Searched tavily successfully")
            
            return normalized
        except Exception as e:
            self.logger.error("Error searching tavily", error=str(e))
            raise BlogGeneratorException("Error searching tavily")
            
        
        
    
    
        
    def _research_node(self,state: State) -> dict:
        
        try:
            self.logger.info("Researching Content")
            # take the first 10 queries from state
            queries = (state.get("queries", []) or [])
            max_results = 6

            raw_results: List[dict] = []

            for q in queries:
                raw_results.extend(self._tavily_search(q, max_results=max_results))

            if not raw_results:
                return {"evidence": []}

            extractor = self.llm.with_structured_output(EvidencePack)
            pack = extractor.invoke(
                [
                    SystemMessage(content=str(RESEARCH_SYSTEM_PROMPT)),
                    HumanMessage(content=f"Raw results:\n{raw_results}"),
                ]
            )

            # Deduplicate by URL
            dedup = {}
            for e in pack.evidence:
                if e.url:
                    dedup[e.url] = e

            self.logger.info("Researched Content successfully")
            return {"evidence": list(dedup.values())}
            
        except Exception as e:
            self.logger.error("Error researching content", error=str(e))
            raise BlogGeneratorException("Error researching content")

       
    
    
    
    def _orchestrator_node(self,state: State) -> dict:
        
        try:
            self.logger.info("Outlining blog post")
            planner = self.llm.with_structured_output(Plan)

            evidence = state.get("evidence", [])
            mode = state.get("mode", "closed_book")

            plan = planner.invoke(
                [
                    SystemMessage(content=str(ORCH_SYSTEM_PROMPT)),
                    HumanMessage(
                        content=(
                            f"Topic: {state['topic']}\n"
                            f"Mode: {mode}\n\n"
                            f"Evidence (ONLY use for fresh claims; may be empty):\n"
                            f"{[e.model_dump() for e in evidence][:16]}"
                        )
                    ),
                ]
            )
           
            self.logger.info("Outlined blog post successfully")
            return {"plan": plan}
            
        except Exception as e:
            self.logger.error("Error outlining blog post", error=str(e))
            raise BlogGeneratorException("Error outlining blog post")
      
    
    
    def _fanout(self, state: State):
        try:
            self.logger.info(
                "Fanout task count",
                count=len(state["plan"].tasks)
            )
            generating_nodes_parallel = [
                Send(
                    "worker",
                    {
                        "task": task.model_dump(),
                        "topic": state["topic"],
                        "mode": state["mode"],
                        "plan": state["plan"].model_dump(),
                        "evidence": [e.model_dump() for e in state.get("evidence", [])],
                    },
                )
                for task in state["plan"].tasks
            ]
            self.logger.info("Fanouted blog post successfully")
            return generating_nodes_parallel
        except Exception as e:
            self.logger.error("Error fanouting blog post", error=str(e))
            raise BlogGeneratorException("Error fanouting blog post")
        
    
    
    
    
    def _worker_node(self, payload: dict) -> dict:
        
        try:
            self.logger.info("Writing each section of the blog post")
            task = Task(**payload["task"])
            plan = Plan(**payload["plan"])
            evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
            topic = payload["topic"]
            mode = payload.get("mode", "closed_book")

            bullets_text = "\n- " + "\n- ".join(task.bullets)

            evidence_text = ""
            if evidence:
                evidence_text = "\n".join(
                    f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
                    for e in evidence[:20]
                )

            section_md = self.llm.invoke(
                [
                    SystemMessage(content=str(WORKER_SYSTEM_PROMPT)),
                    HumanMessage(
                        content=(
                            f"Blog title: {plan.blog_title}\n"
                            f"Audience: {plan.audience}\n"
                            f"Tone: {plan.tone}\n"
                            f"Blog kind: {plan.blog_kind}\n"
                            f"Constraints: {plan.constraints}\n"
                            f"Topic: {topic}\n"
                            f"Mode: {mode}\n\n"
                            f"Section title: {task.title}\n"
                            f"Goal: {task.goal}\n"
                            f"Target words: {task.target_words}\n"
                            f"Tags: {task.tags}\n"
                            f"requires_research: {task.requires_research}\n"
                            f"requires_citations: {task.requires_citations}\n"
                            f"requires_code: {task.requires_code}\n"
                            f"Bullets:{bullets_text}\n\n"
                            f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                        )
                    ),
                ]
            ).content.strip()
            self.logger.info(
                        "Generated section",
                        task_id=task.id,
                        title=task.title,
                        chars=len(section_md)
)
            return {"sections": [(task.id, section_md)]}
        except Exception as e:
            self.logger.error("Error writing each section of the blog post", error=str(e))
            raise BlogGeneratorException("Error writing each section of the blog post")
        
        
    def build_graph(self):
        """
        Construct the Blog Generator Graph.
        """
        
        try:
            self.logger.info("Building blog generator graph")
            builder = StateGraph(State)
            image_generator_graph = MergeImagesWorker(self.llm, self.image_llm).build()
            
            builder.add_node("router",self._router_node)
            builder.add_node("research",self._research_node)
            builder.add_node("orchestrator",self._orchestrator_node)
            builder.add_node("worker",self._worker_node)
            builder.add_node("reducer",image_generator_graph)
            
            builder.add_edge(START,"router")
            builder.add_conditional_edges("router",self._route_next,{"research":"research","orchestrator":"orchestrator"})
            builder.add_edge("research","orchestrator")
            
            builder.add_conditional_edges("orchestrator",self._fanout, ["worker"])
            builder.add_edge("worker","reducer")
            builder.add_edge("reducer",END)
            
            graph = builder.compile(checkpointer=self.memory_saver)
            self.logger.info("Built blog generator graph successfully")
            return graph
        
        except Exception as e:
            self.logger.error("Error building blog generator graph", error=str(e))
            raise BlogGeneratorException("Error building blog generator graph")
        
        
    
    
    


if __name__ == "__main__":
    from datetime import date, timedelta
    try:
        model = ModelLoader()
        llm = model.load_llm()
        image_llm = model.get_image_generator()
        
        reporter = BlogGeneratorWorker(llm, image_llm)
        graph = reporter.build_graph()
        thread = {"configurable": {"thread_id": "1"}}
        topic = "Write a detailed blog on Gradient Boosting"
        
        # for _ in graph.stream({"topic": topic, "max_analysts": 3}, thread, stream_mode="values"):
        #     pass
        def run(topic: str, as_of: Optional[str] = None):
            if as_of is None:
                as_of = date.today().isoformat()

            out = graph.invoke(
                {
                    "topic": topic,
                    "mode": "",
                    "needs_research": False,
                    "queries": [],
                    "evidence": [],
                    "plan": None,
                    "as_of": as_of,
                    "recency_days": 7,
                    "sections": [],
                    "merged_md": "",
                    "md_with_placeholders": "",
                    "image_specs": [],
                    "final": "",
                },
                thread
            )

            return out
        
        run("Write a detailed blog on Gradient Boosting")
        reporter.logger.info("Blog generated successfully")
    except Exception as e:
        reporter.logger.error("Error generating blog", error=str(e))
        raise BlogGeneratorException("Error generating blog")
    
        
        
        