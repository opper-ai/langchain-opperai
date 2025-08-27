"""
Multi-Researcher LangGraph + Opper Integration Demo

This demo showcases a sophisticated multi-agent research workflow where multiple
specialized researchers collaborate to produce a comprehensive final report.

Features demonstrated:
1. Multiple specialized researcher agents working in parallel
2. Structured input and output using Pydantic models
3. Proper trace management across all workflow steps
4. State management with researcher collaboration
5. Final synthesis node that combines all research
6. Complete LangGraph workflow with Opper integration
"""

import os
from typing import List, Dict, Optional, Any, TypedDict, Annotated
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from langchain_opper import OpperProvider, OpperChatModel

load_dotenv()
console = Console()


# ==================== INPUT SCHEMAS ====================

class ResearchRequest(BaseModel):
    """Structured input for research requests."""
    topic: str = Field(description="The main topic to research")
    focus_area: str = Field(description="Specific focus area for this researcher")
    research_depth: str = Field(description="Depth of research required")
    perspective: str = Field(description="Research perspective or angle")
    time_horizon: str = Field(description="Time horizon for analysis")


# ==================== OUTPUT SCHEMAS ====================

class ResearchFindings(BaseModel):
    """Research findings from a single researcher."""
    thoughts: str = Field(description="Research analysis process")
    researcher_type: str = Field(description="Type/specialty of this researcher")
    key_findings: List[str] = Field(description="Main research discoveries")
    sources: List[str] = Field(description="Information sources used")
    confidence_score: float = Field(description="Confidence in findings (0-1)", ge=0, le=1)
    research_gaps: List[str] = Field(description="Areas that need more investigation")
    recommendations: List[str] = Field(description="Specific recommendations from this research")


class SynthesisReport(BaseModel):
    """Final comprehensive report synthesizing all research."""
    thoughts: str = Field(description="Synthesis reasoning process")
    executive_summary: str = Field(description="High-level executive summary")
    key_insights: List[str] = Field(description="Key insights across all research")
    detailed_analysis: str = Field(description="Comprehensive analysis section")
    researcher_contributions: Dict[str, str] = Field(description="Summary of each researcher's contributions")
    final_recommendations: List[str] = Field(description="Final prioritized recommendations")
    confidence_assessment: str = Field(description="Overall confidence in findings")
    next_steps: List[str] = Field(description="Suggested implementation steps")


# ==================== LANGGRAPH STATE ====================

class MultiResearchState(TypedDict):
    """State for the multi-researcher workflow."""
    messages: Annotated[List, add_messages]
    query: str
    provider: OpperProvider
    
    # Research findings from each specialist
    market_research: Optional[ResearchFindings]
    technical_research: Optional[ResearchFindings] 
    competitive_research: Optional[ResearchFindings]
    trend_research: Optional[ResearchFindings]
    
    # Final synthesis
    synthesis_report: Optional[SynthesisReport]
    
    metadata: Dict[str, Any]


# ==================== RESEARCHER NODES ====================

def market_researcher_node(state: MultiResearchState) -> MultiResearchState:
    """Market research specialist focusing on market dynamics and opportunities."""
    console.print("🏢 [bold blue]Market Research Specialist[/bold blue]")
    
    provider = state["provider"]
    
    # Create specialized market research model
    market_model = provider.create_structured_model(
        task_name="market_research",
        instructions=(
            "You are a market research specialist. Focus on market dynamics, opportunities, "
            "customer needs, market size, growth potential, and business viability. "
            "Provide data-driven insights about market conditions and opportunities."
        ),
        output_schema=ResearchFindings
    )
    
    # Create structured input for market research
    research_request = ResearchRequest(
        topic=state['query'],
        focus_area="market dynamics and business opportunities",
        research_depth="comprehensive",
        perspective="business and market analysis",
        time_horizon="current market and 2-3 year outlook"
    )
    
    console.print("   📋 [cyan]Market Research Focus:[/cyan]")
    console.print(f"      • Market dynamics and opportunities")
    console.print(f"      • Customer needs and segments")
    console.print(f"      • Business viability assessment")
    
    # Create research message
    message = HumanMessage(
        content=f"Conduct comprehensive market research on: {state['query']}",
        additional_kwargs={
            "research_request": research_request.model_dump(),
            "researcher_type": "market_specialist"
        }
    )
    
    # Get research findings
    result = market_model.invoke([message])
    findings = result.additional_kwargs.get("parsed")
    span_id = result.additional_kwargs.get("span_id")
    
    console.print(f"   📊 [green]Market research completed[/green] (Span: {span_id})")
    
    return {
        "market_research": findings,
        "messages": state["messages"] + [result],
        "metadata": {
            **state.get("metadata", {}),
            "market_span_id": span_id,
            "market_completed": True
        }
    }


def technical_researcher_node(state: MultiResearchState) -> MultiResearchState:
    """Technical research specialist focusing on implementation and technical feasibility."""
    console.print("⚙️ [bold green]Technical Research Specialist[/bold green]")
    
    provider = state["provider"]
    
    # Create specialized technical research model
    tech_model = provider.create_structured_model(
        task_name="technical_research",
        instructions=(
            "You are a technical research specialist. Focus on technical implementation, "
            "infrastructure requirements, technology stack, scalability, security, "
            "and technical feasibility. Provide technical depth and implementation insights."
        ),
        output_schema=ResearchFindings
    )
    
    research_request = ResearchRequest(
        topic=state['query'],
        focus_area="technical implementation and feasibility",
        research_depth="detailed technical analysis",
        perspective="technical architecture and engineering",
        time_horizon="implementation timeline and technical roadmap"
    )
    
    console.print("   📋 [cyan]Technical Research Focus:[/cyan]")
    console.print(f"      • Technical feasibility and architecture")
    console.print(f"      • Implementation requirements")
    console.print(f"      • Scalability and security considerations")
    
    message = HumanMessage(
        content=f"Conduct technical feasibility research on: {state['query']}",
        additional_kwargs={
            "research_request": research_request.model_dump(),
            "researcher_type": "technical_specialist"
        }
    )
    
    result = tech_model.invoke([message])
    findings = result.additional_kwargs.get("parsed")
    span_id = result.additional_kwargs.get("span_id")
    
    console.print(f"   📊 [green]Technical research completed[/green] (Span: {span_id})")
    
    return {
        "technical_research": findings,
        "messages": state["messages"] + [result],
        "metadata": {
            **state.get("metadata", {}),
            "technical_span_id": span_id,
            "technical_completed": True
        }
    }


def competitive_researcher_node(state: MultiResearchState) -> MultiResearchState:
    """Competitive research specialist focusing on competitive landscape and positioning."""
    console.print("🏆 [bold yellow]Competitive Research Specialist[/bold yellow]")
    
    provider = state["provider"]
    
    competitive_model = provider.create_structured_model(
        task_name="competitive_research",
        instructions=(
            "You are a competitive intelligence specialist. Focus on competitive landscape, "
            "competitor analysis, market positioning, competitive advantages, threats, "
            "and strategic positioning opportunities."
        ),
        output_schema=ResearchFindings
    )
    
    research_request = ResearchRequest(
        topic=state['query'],
        focus_area="competitive landscape and positioning",
        research_depth="comprehensive competitive analysis",
        perspective="competitive intelligence and strategic positioning",
        time_horizon="current competitive state and strategic outlook"
    )
    
    console.print("   📋 [cyan]Competitive Research Focus:[/cyan]")
    console.print(f"      • Competitive landscape mapping")
    console.print(f"      • Strategic positioning analysis")
    console.print(f"      • Competitive advantages and threats")
    
    message = HumanMessage(
        content=f"Conduct competitive landscape analysis for: {state['query']}",
        additional_kwargs={
            "research_request": research_request.model_dump(),
            "researcher_type": "competitive_specialist"
        }
    )
    
    result = competitive_model.invoke([message])
    findings = result.additional_kwargs.get("parsed")
    span_id = result.additional_kwargs.get("span_id")
    
    console.print(f"   📊 [green]Competitive research completed[/green] (Span: {span_id})")
    
    return {
        "competitive_research": findings,
        "messages": state["messages"] + [result],
        "metadata": {
            **state.get("metadata", {}),
            "competitive_span_id": span_id,
            "competitive_completed": True
        }
    }


def trend_researcher_node(state: MultiResearchState) -> MultiResearchState:
    """Trend research specialist focusing on future trends and emerging opportunities."""
    console.print("🔮 [bold magenta]Trend Research Specialist[/bold magenta]")
    
    provider = state["provider"]
    
    trend_model = provider.create_structured_model(
        task_name="trend_research", 
        instructions=(
            "You are a trend research specialist. Focus on emerging trends, future outlook, "
            "innovation opportunities, industry evolution, and long-term strategic implications. "
            "Provide forward-looking insights and trend analysis."
        ),
        output_schema=ResearchFindings
    )
    
    research_request = ResearchRequest(
        topic=state['query'],
        focus_area="emerging trends and future opportunities",
        research_depth="trend analysis and future forecasting",
        perspective="innovation and long-term strategic outlook",
        time_horizon="emerging trends and 3-5 year strategic outlook"
    )
    
    console.print("   📋 [cyan]Trend Research Focus:[/cyan]")
    console.print(f"      • Emerging trends and innovations")
    console.print(f"      • Future market evolution")
    console.print(f"      • Long-term strategic opportunities")
    
    message = HumanMessage(
        content=f"Conduct trend analysis and future outlook research for: {state['query']}",
        additional_kwargs={
            "research_request": research_request.model_dump(),
            "researcher_type": "trend_specialist"
        }
    )
    
    result = trend_model.invoke([message])
    findings = result.additional_kwargs.get("parsed")
    span_id = result.additional_kwargs.get("span_id")
    
    console.print(f"   📊 [green]Trend research completed[/green] (Span: {span_id})")
    
    return {
        "trend_research": findings,
        "messages": state["messages"] + [result],
        "metadata": {
            **state.get("metadata", {}),
            "trend_span_id": span_id,
            "trend_completed": True
        }
    }


def synthesis_node(state: MultiResearchState) -> MultiResearchState:
    """Synthesis specialist that combines all research into a comprehensive report."""
    console.print("🎯 [bold red]Research Synthesis Specialist[/bold red]")
    
    provider = state["provider"]
    
    synthesis_model = provider.create_structured_model(
        task_name="research_synthesis",
        instructions=(
            "You are a research synthesis specialist. Your role is to analyze and synthesize "
            "findings from multiple research specialists into a comprehensive, coherent report. "
            "Identify key insights, resolve conflicts, highlight synergies, and provide "
            "executive-level recommendations based on the collective research."
        ),
        output_schema=SynthesisReport
    )
    
    # Compile all research findings
    research_data = {
        "market_research": state["market_research"],
        "technical_research": state["technical_research"], 
        "competitive_research": state["competitive_research"],
        "trend_research": state["trend_research"]
    }
    
    console.print("   📋 [cyan]Synthesizing Research From:[/cyan]")
    for research_type, findings in research_data.items():
        if findings:
            console.print(f"      • {research_type.replace('_', ' ').title()}: {len(findings.key_findings)} findings")
    
    # Create comprehensive synthesis prompt
    synthesis_prompt = f"""
    Synthesize comprehensive research findings for: {state['query']}
    
    MARKET RESEARCH FINDINGS:
    Key Findings: {', '.join(research_data['market_research'].key_findings)}
    Recommendations: {', '.join(research_data['market_research'].recommendations)}
    Confidence: {research_data['market_research'].confidence_score}
    
    TECHNICAL RESEARCH FINDINGS:
    Key Findings: {', '.join(research_data['technical_research'].key_findings)}
    Recommendations: {', '.join(research_data['technical_research'].recommendations)}
    Confidence: {research_data['technical_research'].confidence_score}
    
    COMPETITIVE RESEARCH FINDINGS:
    Key Findings: {', '.join(research_data['competitive_research'].key_findings)}
    Recommendations: {', '.join(research_data['competitive_research'].recommendations)}
    Confidence: {research_data['competitive_research'].confidence_score}
    
    TREND RESEARCH FINDINGS:
    Key Findings: {', '.join(research_data['trend_research'].key_findings)}
    Recommendations: {', '.join(research_data['trend_research'].recommendations)}
    Confidence: {research_data['trend_research'].confidence_score}
    
    Please synthesize these findings into a comprehensive executive report.
    """
    
    message = HumanMessage(
        content=synthesis_prompt,
        additional_kwargs={
            "synthesis_request": {
                "topic": state['query'],
                "research_inputs": len([r for r in research_data.values() if r]),
                "synthesis_type": "executive_report"
            }
        }
    )
    
    result = synthesis_model.invoke([message])
    synthesis = result.additional_kwargs.get("parsed")
    span_id = result.additional_kwargs.get("span_id")
    
    console.print(f"   📊 [green]Research synthesis completed[/green] (Span: {span_id})")
    
    return {
        "synthesis_report": synthesis,
        "messages": state["messages"] + [result],
        "metadata": {
            **state.get("metadata", {}),
            "synthesis_span_id": span_id,
            "synthesis_completed": True
        }
    }


# ==================== WORKFLOW BUILDER ====================

def create_multi_researcher_workflow() -> StateGraph:
    """Create the multi-researcher LangGraph workflow."""
    
    workflow = StateGraph(MultiResearchState)
    
    # Add all researcher nodes
    workflow.add_node("market_researcher", market_researcher_node)
    workflow.add_node("technical_researcher", technical_researcher_node)
    workflow.add_node("competitive_researcher", competitive_researcher_node)
    workflow.add_node("trend_researcher", trend_researcher_node)
    workflow.add_node("synthesis", synthesis_node)
    
    # Sequential execution: each researcher builds on the previous
    workflow.set_entry_point("market_researcher")
    workflow.add_edge("market_researcher", "technical_researcher")
    workflow.add_edge("technical_researcher", "competitive_researcher")
    workflow.add_edge("competitive_researcher", "trend_researcher")
    workflow.add_edge("trend_researcher", "synthesis")
    
    # Synthesis is the final step
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()


# ==================== DEMO EXECUTION ====================

def run_multi_researcher_demo():
    """Run the multi-researcher collaboration demo."""
    
    # Demo topic
    topic = "AI-powered personalized learning platforms for corporate training"
    
    console.print(Panel.fit(
        f"[bold blue]🔬 Multi-Researcher Collaboration Demo[/bold blue]\n\n"
        f"[bold yellow]Research Topic:[/bold yellow]\n{topic}\n\n"
        f"[bold cyan]Research Team:[/bold cyan]\n"
        f"• 🏢 Market Research Specialist\n"
        f"• ⚙️ Technical Research Specialist\n" 
        f"• 🏆 Competitive Research Specialist\n"
        f"• 🔮 Trend Research Specialist\n"
        f"• 🎯 Research Synthesis Specialist",
        border_style="blue"
    ))
    
    # Check for API key
    if not os.getenv("OPPER_API_KEY"):
        console.print(Panel(
            "[red]❌ OPPER_API_KEY not found![/red]\n\n"
            "Please set your Opper API key:\n"
            "[cyan]export OPPER_API_KEY='your-api-key'[/cyan]",
            border_style="red"
        ))
        return
    
    # Initialize provider and start trace
    provider = OpperProvider()
    trace_id = provider.start_trace("multi_researcher_workflow", topic)
    console.print(f"🔍 [bold cyan]Started multi-researcher trace:[/bold cyan] {trace_id}")
    
    try:
        # Create and run workflow
        workflow = create_multi_researcher_workflow()
        
        # Initial state
        initial_state = {
            "messages": [],
            "query": topic,
            "provider": provider,
            "market_research": None,
            "technical_research": None,
            "competitive_research": None,
            "trend_research": None,
            "synthesis_report": None,
            "metadata": {"workflow_started": True, "trace_id": trace_id}
        }
        
        # Run workflow with progress
        console.print("\n🚀 [bold green]Running multi-researcher workflow...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Coordinating research team...", total=None)
            final_state = workflow.invoke(initial_state)
            progress.update(task, description="✅ Research completed!")
        
        # Display comprehensive results
        display_multi_research_results(final_state)
        
        # Display trace summary
        display_trace_summary(final_state, trace_id)
        
        # End trace
        provider.end_trace("Multi-researcher workflow completed successfully")
        console.print(f"✅ [bold green]Completed multi-researcher trace:[/bold green] {trace_id}")
        
    except Exception as e:
        console.print(f"[red]Error running multi-researcher workflow: {e}[/red]")
        provider.end_trace(f"Workflow failed: {e}")
        raise


def display_multi_research_results(state: MultiResearchState):
    """Display comprehensive results from all researchers."""
    
    console.print("\n" + "="*80)
    console.print("[bold blue]📊 MULTI-RESEARCHER COLLABORATION RESULTS[/bold blue]")
    console.print("="*80)
    
    # Research team contributions
    researchers = [
        ("market_research", "🏢 Market Research", "blue"),
        ("technical_research", "⚙️ Technical Research", "green"), 
        ("competitive_research", "🏆 Competitive Research", "yellow"),
        ("trend_research", "🔮 Trend Research", "magenta")
    ]
    
    for research_key, title, color in researchers:
        research = state.get(research_key)
        if research:
            table = Table(title=title)
            table.add_column("Aspect", style="cyan")
            table.add_column("Findings", style="white")
            
            table.add_row("Key Findings", "\n".join(f"• {finding}" for finding in research.key_findings[:3]))
            table.add_row("Confidence", f"{research.confidence_score:.2f}")
            table.add_row("Recommendations", "\n".join(f"• {rec}" for rec in research.recommendations[:2]))
            
            console.print(table)
            console.print()
    
    # Synthesis Report
    if state["synthesis_report"]:
        synthesis = state["synthesis_report"]
        
        console.print(Panel(
            f"[bold blue]🎯 EXECUTIVE SYNTHESIS REPORT[/bold blue]\n\n"
            f"[bold yellow]Executive Summary[/bold yellow]\n{synthesis.executive_summary}\n\n"
            f"[bold green]Key Insights[/bold green]\n" +
            "\n".join(f"• {insight}" for insight in synthesis.key_insights) + "\n\n"
            f"[bold cyan]Final Recommendations[/bold cyan]\n" +
            "\n".join(f"• {rec}" for rec in synthesis.final_recommendations) + "\n\n"
            f"[bold magenta]Confidence Assessment[/bold magenta]\n{synthesis.confidence_assessment}",
            title="📋 COMPREHENSIVE RESEARCH SYNTHESIS",
            border_style="red"
        ))


def display_trace_summary(state: MultiResearchState, trace_id: str):
    """Display comprehensive trace information."""
    metadata = state.get("metadata", {})
    
    trace_table = Table(title="🔍 Multi-Researcher Trace Summary")
    trace_table.add_column("Component", style="cyan")
    trace_table.add_column("Span ID", style="yellow")
    trace_table.add_column("Status", style="green")
    
    trace_table.add_row("Parent Trace", trace_id, "✅ Active")
    
    researchers = [
        ("market_span_id", "Market Researcher"),
        ("technical_span_id", "Technical Researcher"),
        ("competitive_span_id", "Competitive Researcher"),
        ("trend_span_id", "Trend Researcher"),
        ("synthesis_span_id", "Synthesis Specialist")
    ]
    
    for span_key, component in researchers:
        span_id = metadata.get(span_key)
        if span_id:
            trace_table.add_row(component, span_id, "✅ Completed")
    
    console.print(trace_table)
    console.print()


if __name__ == "__main__":
    run_multi_researcher_demo()