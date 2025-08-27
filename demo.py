"""
Real-World kwargs Example: AI-Powered Business Consulting Platform

This comprehensive demo showcases a business consulting platform that uses
HumanMessage additional_kwargs to pass rich context data for sophisticated
business analysis across multiple domains.

Scenarios covered:
1. Market Entry Analysis with Financial Constraints
2. Product Launch Strategy with Competitive Intelligence
3. M&A Due Diligence with Risk Assessment
4. Digital Transformation Planning with Legacy Constraints
"""

import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated

from langchain_opper import OpperProvider

load_dotenv()
console = Console()


# ==================== SCHEMAS ====================

class MarketEntryAnalysis(BaseModel):
    """Market entry analysis with financial constraints."""
    thoughts: str = Field(description="Strategic analysis process")
    market_attractiveness: float = Field(description="Market attractiveness score 0-1")
    entry_barriers: List[str] = Field(description="Key barriers to entry")
    recommended_strategy: str = Field(description="Recommended market entry approach")
    investment_requirements: str = Field(description="Estimated investment needed")
    timeline_to_breakeven: str = Field(description="Expected time to breakeven")
    success_probability: float = Field(description="Success probability 0-1")
    key_risks: List[str] = Field(description="Primary risk factors")
    next_steps: List[str] = Field(description="Immediate action items")


class ProductLaunchStrategy(BaseModel):
    """Product launch strategy with competitive analysis."""
    thoughts: str = Field(description="Product strategy reasoning")
    positioning_statement: str = Field(description="Core product positioning")
    target_segments: List[str] = Field(description="Primary target segments")
    pricing_strategy: str = Field(description="Recommended pricing approach")
    launch_channels: List[str] = Field(description="Optimal launch channels")
    competitive_advantages: List[str] = Field(description="Key differentiators")
    marketing_budget_allocation: Dict[str, str] = Field(description="Budget allocation by channel")
    launch_timeline: str = Field(description="Recommended launch timeline")
    success_metrics: List[str] = Field(description="Key performance indicators")


class DueDiligenceAssessment(BaseModel):
    """M&A due diligence assessment."""
    thoughts: str = Field(description="Due diligence analysis process")
    overall_valuation: str = Field(description="Estimated company valuation")
    financial_health_score: float = Field(description="Financial health score 0-1")
    strategic_fit_score: float = Field(description="Strategic fit score 0-1")
    key_strengths: List[str] = Field(description="Company's key strengths")
    critical_risks: List[str] = Field(description="Major risk factors")
    integration_complexity: str = Field(description="Post-acquisition integration assessment")
    synergy_potential: str = Field(description="Expected synergies")
    recommendation: str = Field(description="Final acquisition recommendation")
    deal_breakers: List[str] = Field(description="Potential deal-breaking issues")


class DigitalTransformationPlan(BaseModel):
    """Digital transformation strategy."""
    thoughts: str = Field(description="Transformation planning reasoning")
    current_maturity_level: str = Field(description="Current digital maturity assessment")
    transformation_roadmap: List[str] = Field(description="Phased transformation steps")
    technology_stack: List[str] = Field(description="Recommended technology solutions")
    change_management_approach: str = Field(description="Change management strategy")
    investment_schedule: str = Field(description="Investment timeline and amounts")
    roi_projections: str = Field(description="Expected return on investment")
    implementation_risks: List[str] = Field(description="Key implementation risks")
    success_factors: List[str] = Field(description="Critical success factors")


# ==================== WORKFLOW STATE ====================

class ConsultingWorkflowState(TypedDict):
    """State for business consulting workflow."""
    messages: Annotated[list, add_messages]
    analysis_type: str
    client_context: Dict[str, Any]
    market_analysis: Optional[MarketEntryAnalysis]
    product_strategy: Optional[ProductLaunchStrategy]
    due_diligence: Optional[DueDiligenceAssessment]
    digital_strategy: Optional[DigitalTransformationPlan]
    final_recommendation: str


# ==================== CONSULTING SCENARIOS ====================

def scenario_market_entry(provider: OpperProvider):
    """Market entry analysis for international expansion."""
    console.print(Panel("🌍 Scenario 1: International Market Entry Analysis", style="bold blue"))
    
    # Create specialized market entry analyst
    market_analyst = provider.create_chat_model(
        task_name="international_market_analyst",
        instructions="""You are a senior international market analyst specializing in market entry strategies.

Analyze the market entry opportunity using the provided context:
- input: The main analysis request
- context: Contains detailed market and business context including:
  - target_market: The market to enter
  - company_size: Current company scale
  - industry: Business sector
  - budget: Available investment budget
  - timeline: Expansion timeline
  - current_markets: Existing market presence
  - competitive_position: Current competitive standing
  - risk_tolerance: Company's risk appetite

Provide comprehensive market entry analysis with actionable recommendations.""",
        model_name="anthropic/claude-3.5-sonnet"
    ).with_structured_output(schema=MarketEntryAnalysis)
    
    # Rich business context through kwargs
    market_entry_request = HumanMessage(
        content="Analyze our expansion opportunity into the European fintech market, specifically focusing on Germany and France for our digital banking platform.",
        additional_kwargs={
            "target_market": "European Union (Germany, France)",
            "company_size": "Series B fintech startup, 150 employees",
            "industry": "Digital banking and financial services",
            "budget": "$15M allocated for European expansion",
            "timeline": "18-month expansion timeline",
            "current_markets": ["United States", "Canada", "United Kingdom"],
            "competitive_position": "Strong in mobile-first banking, weak in enterprise",
            "risk_tolerance": "Medium - need 70% success probability",
            "regulatory_requirements": "Must comply with GDPR, PSD2, local banking regulations",
            "team_readiness": "Limited European expertise, strong tech team",
            "product_localization": "Platform supports multi-currency, needs local payment methods"
        }
    )
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Analyzing market entry opportunity...", total=None)
        result = market_analyst.invoke([market_entry_request])
    
    analysis = result.additional_kwargs.get("parsed")
    if analysis:
        display_market_analysis(analysis)
        
        # Add metrics (using provider's current trace)
        current_trace = provider.current_trace_id
        if current_trace:
            provider.add_metric(current_trace, "market_attractiveness", analysis.market_attractiveness, "Market attractiveness score")
            provider.add_metric(current_trace, "success_probability", analysis.success_probability, "Success probability assessment")
    
    return analysis


def scenario_product_launch(provider: OpperProvider):
    """Product launch strategy with competitive intelligence."""
    console.print(Panel("🚀 Scenario 2: AI Product Launch Strategy", style="bold green"))
    
    # Create product strategy consultant
    product_strategist = provider.create_chat_model(
        task_name="ai_product_strategist",
        instructions="""You are a senior product strategist specializing in AI product launches.

Develop a comprehensive launch strategy using the provided context:
- input: The product launch challenge
- context: Contains detailed product and market context including:
  - product_type: Type of AI product
  - target_customers: Primary customer segments
  - development_stage: Current product readiness
  - competitive_landscape: Key competitors and positioning
  - budget: Marketing and launch budget
  - team_size: Product and marketing team capacity
  - unique_features: Key differentiating features
  - pricing_model: Revenue model approach

Create a detailed go-to-market strategy with tactical recommendations.""",
        model_name="anthropic/claude-3.5-sonnet"
    ).with_structured_output(schema=ProductLaunchStrategy)
    
    # Comprehensive product context
    product_launch_request = HumanMessage(
        content="Develop a go-to-market strategy for our new AI-powered customer service automation platform targeting mid-market SaaS companies.",
        additional_kwargs={
            "product_type": "AI customer service automation platform",
            "target_customers": "Mid-market SaaS companies (50-500 employees)",
            "development_stage": "Beta testing complete, production-ready",
            "competitive_landscape": {
                "direct_competitors": ["Zendesk AI", "Intercom Resolution Bot", "Ada"],
                "indirect_competitors": ["Traditional help desk software", "Custom chatbot solutions"],
                "competitive_advantages": ["Superior NLP accuracy", "Easy integration", "Transparent pricing"]
            },
            "budget": "$2M marketing budget for 12-month launch",
            "team_size": "15-person product team, 8-person marketing team",
            "unique_features": [
                "99.2% intent recognition accuracy",
                "5-minute integration setup",
                "Multi-language support (12 languages)",
                "Real-time sentiment analysis",
                "Seamless human handoff"
            ],
            "pricing_model": "Per-conversation pricing starting at $0.50/resolved conversation",
            "customer_validation": "12 beta customers, 94% satisfaction rate",
            "technical_readiness": "99.9% uptime, sub-200ms response time",
            "sales_cycle": "Typically 30-45 days for mid-market",
            "churn_benchmark": "Industry average 8% monthly churn"
        }
    )
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Developing product launch strategy...", total=None)
        result = product_strategist.invoke([product_launch_request])
    
    strategy = result.additional_kwargs.get("parsed")
    if strategy:
        display_product_strategy(strategy)
        
        # Add metrics (using provider's current trace)
        current_trace = provider.current_trace_id
        if current_trace:
            provider.add_metric(current_trace, "strategy_confidence", 0.85, "Product strategy confidence level")
    
    return strategy


def scenario_due_diligence(provider: OpperProvider):
    """M&A due diligence assessment."""
    console.print(Panel("🔍 Scenario 3: M&A Due Diligence Assessment", style="bold magenta"))
    
    # Create M&A analyst
    ma_analyst = provider.create_chat_model(
        task_name="ma_due_diligence_specialist",
        instructions="""You are a senior M&A analyst conducting due diligence assessments.

Perform comprehensive due diligence analysis using the provided context:
- input: The acquisition assessment request
- context: Contains detailed target company information including:
  - target_company: Company being evaluated
  - financial_metrics: Key financial data
  - business_model: Revenue and operating model
  - market_position: Competitive standing
  - technology_stack: Technical infrastructure
  - team_assessment: Leadership and talent evaluation
  - strategic_rationale: Acquisition reasoning
  - deal_parameters: Transaction structure

Provide thorough due diligence assessment with clear recommendation.""",
        model_name="anthropic/claude-3.5-sonnet"
    ).with_structured_output(schema=DueDiligenceAssessment)
    
    # Detailed acquisition context
    due_diligence_request = HumanMessage(
        content="Conduct due diligence assessment for the proposed acquisition of DataFlow Analytics, an AI-powered business intelligence startup.",
        additional_kwargs={
            "target_company": "DataFlow Analytics - AI business intelligence platform",
            "financial_metrics": {
                "annual_revenue": "$8.5M ARR",
                "growth_rate": "180% YoY revenue growth",
                "gross_margin": "85%",
                "burn_rate": "$1.2M monthly",
                "runway": "14 months remaining",
                "customer_count": "145 enterprise customers",
                "average_deal_size": "$65K annually"
            },
            "business_model": "SaaS subscription with usage-based pricing tiers",
            "market_position": "Leader in AI-powered data visualization for manufacturing",
            "technology_stack": {
                "ai_capabilities": "Proprietary ML models for predictive analytics",
                "infrastructure": "Cloud-native on AWS",
                "scalability": "Proven to handle 10TB+ data processing",
                "ip_portfolio": "3 pending patents, 15 proprietary algorithms"
            },
            "team_assessment": {
                "leadership": "Strong founding team with domain expertise",
                "engineering": "12 senior engineers, low turnover",
                "sales": "Proven enterprise sales process",
                "customer_success": "95% customer satisfaction score"
            },
            "strategic_rationale": "Expands our AI capabilities into manufacturing vertical",
            "deal_parameters": {
                "valuation": "$75M proposed acquisition price",
                "structure": "60% cash, 40% stock with earnout",
                "earnout_terms": "Additional $15M based on revenue milestones",
                "key_person_retention": "2-year retention agreements for founders"
            },
            "synergies_potential": {
                "revenue_synergies": "Cross-selling to existing customer base",
                "cost_synergies": "Shared infrastructure and sales operations",
                "technology_synergies": "Enhanced AI capabilities across all products"
            },
            "risk_factors": [
                "Key customer concentration (top 3 customers = 40% revenue)",
                "Regulatory compliance in manufacturing sector",
                "Competition from Microsoft and Tableau"
            ]
        }
    )
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Conducting due diligence assessment...", total=None)
        result = ma_analyst.invoke([due_diligence_request])
    
    assessment = result.additional_kwargs.get("parsed")
    if assessment:
        display_due_diligence(assessment)
        
        # Add metrics (using provider's current trace)
        current_trace = provider.current_trace_id
        if current_trace:
            provider.add_metric(current_trace, "financial_health", assessment.financial_health_score, "Target company financial health")
            provider.add_metric(current_trace, "strategic_fit", assessment.strategic_fit_score, "Strategic fit assessment")
    
    return assessment


def scenario_digital_transformation(provider: OpperProvider):
    """Digital transformation planning with legacy constraints."""
    console.print(Panel("💻 Scenario 4: Enterprise Digital Transformation", style="bold cyan"))
    
    # Create digital transformation consultant
    transformation_consultant = provider.create_chat_model(
        task_name="digital_transformation_strategist",
        instructions="""You are a senior digital transformation consultant specializing in large enterprise transformations.

Develop comprehensive transformation strategy using the provided context:
- input: The transformation challenge
- context: Contains detailed organizational context including:
  - company_profile: Organization details
  - current_state: Existing technology landscape
  - business_objectives: Transformation goals
  - constraints: Limitations and challenges
  - budget: Investment capacity
  - timeline: Transformation timeline
  - stakeholders: Key decision makers
  - change_readiness: Organizational change capacity

Create a practical, phased transformation roadmap with clear milestones.""",
        model_name="anthropic/claude-3.5-sonnet"
    ).with_structured_output(schema=DigitalTransformationPlan)
    
    # Complex enterprise context
    transformation_request = HumanMessage(
        content="Design a comprehensive digital transformation strategy for our 150-year-old manufacturing company to modernize operations and become data-driven.",
        additional_kwargs={
            "company_profile": {
                "industry": "Heavy machinery manufacturing",
                "size": "5,000 employees across 12 global facilities",
                "revenue": "$2.8B annually",
                "established": "1874 - traditional manufacturing heritage"
            },
            "current_state": {
                "technology_landscape": "Mix of legacy systems from 1990s-2010s",
                "erp_system": "SAP R/3 implemented in 2005, heavily customized",
                "data_infrastructure": "Siloed databases, limited integration",
                "automation_level": "30% of production processes automated",
                "digital_skills": "Limited across workforce, strong engineering talent",
                "customer_interaction": "Primarily offline sales and support"
            },
            "business_objectives": [
                "Increase operational efficiency by 25%",
                "Reduce time-to-market for new products by 40%",
                "Implement predictive maintenance across all facilities",
                "Enable real-time supply chain visibility",
                "Launch digital customer portal and e-commerce platform"
            ],
            "constraints": {
                "legacy_systems": "Cannot replace ERP - must integrate around it",
                "regulatory": "Strict safety and quality regulations in aerospace/defense",
                "workforce": "Aging workforce with limited digital skills",
                "capital": "Conservative capital allocation - ROI required within 3 years",
                "operational": "Cannot disrupt production - 99.5% uptime requirement"
            },
            "budget": "$50M over 4 years for digital transformation",
            "timeline": "Must show measurable results within 18 months",
            "stakeholders": {
                "ceo": "Committed but risk-averse",
                "cto": "Recently hired from tech industry",
                "operations": "Skeptical of change, focused on reliability",
                "finance": "Demanding clear ROI justification",
                "hr": "Concerned about workforce adaptation"
            },
            "change_readiness": {
                "leadership_support": "Strong from C-suite",
                "middle_management": "Mixed - some resistance",
                "frontline": "Low digital literacy but willing to learn",
                "culture": "Traditional, risk-averse, quality-focused"
            },
            "competitive_pressure": "Newer manufacturers leveraging Industry 4.0 technologies",
            "customer_expectations": "Demanding digital interfaces and real-time visibility"
        }
    )
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Designing digital transformation strategy...", total=None)
        result = transformation_consultant.invoke([transformation_request])
    
    plan = result.additional_kwargs.get("parsed")
    if plan:
        display_transformation_plan(plan)
        
        # Add metrics (using provider's current trace)
        current_trace = provider.current_trace_id
        if current_trace:
            provider.add_metric(current_trace, "transformation_complexity", 0.8, "Transformation complexity assessment")
    
    return plan


# ==================== DISPLAY FUNCTIONS ====================

def display_market_analysis(analysis: MarketEntryAnalysis):
    """Display market entry analysis results."""
    console.print(Panel.fit(
        f"""[bold blue]Market Entry Analysis Results[/bold blue]

🎯 [bold]Market Attractiveness:[/bold] [green]{analysis.market_attractiveness:.1%}[/green]
🎲 [bold]Success Probability:[/bold] [green]{analysis.success_probability:.1%}[/green]

🚧 [bold]Entry Barriers:[/bold]
{chr(10).join(f'  • {barrier}' for barrier in analysis.entry_barriers)}

💡 [bold]Recommended Strategy:[/bold]
{analysis.recommended_strategy}

💰 [bold]Investment Requirements:[/bold] {analysis.investment_requirements}
⏱️  [bold]Timeline to Breakeven:[/bold] {analysis.timeline_to_breakeven}

⚠️  [bold]Key Risks:[/bold]
{chr(10).join(f'  • {risk}' for risk in analysis.key_risks)}

🚀 [bold]Next Steps:[/bold]
{chr(10).join(f'  • {step}' for step in analysis.next_steps)}""",
        title="🌍 Market Entry",
        border_style="blue"
    ))


def display_product_strategy(strategy: ProductLaunchStrategy):
    """Display product launch strategy results."""
    console.print(Panel.fit(
        f"""[bold green]Product Launch Strategy[/bold green]

🎯 [bold]Positioning:[/bold] {strategy.positioning_statement}

👥 [bold]Target Segments:[/bold]
{chr(10).join(f'  • {segment}' for segment in strategy.target_segments)}

💰 [bold]Pricing Strategy:[/bold] {strategy.pricing_strategy}

📢 [bold]Launch Channels:[/bold]
{chr(10).join(f'  • {channel}' for channel in strategy.launch_channels)}

⭐ [bold]Competitive Advantages:[/bold]
{chr(10).join(f'  • {advantage}' for advantage in strategy.competitive_advantages)}

📊 [bold]Budget Allocation:[/bold]
{chr(10).join(f'  • {channel}: {allocation}' for channel, allocation in strategy.marketing_budget_allocation.items())}

📅 [bold]Timeline:[/bold] {strategy.launch_timeline}

📈 [bold]Success Metrics:[/bold]
{chr(10).join(f'  • {metric}' for metric in strategy.success_metrics)}""",
        title="🚀 Product Launch",
        border_style="green"
    ))


def display_due_diligence(assessment: DueDiligenceAssessment):
    """Display due diligence assessment results."""
    financial_color = "green" if assessment.financial_health_score >= 0.7 else "yellow" if assessment.financial_health_score >= 0.5 else "red"
    strategic_color = "green" if assessment.strategic_fit_score >= 0.7 else "yellow" if assessment.strategic_fit_score >= 0.5 else "red"
    
    console.print(Panel.fit(
        f"""[bold magenta]M&A Due Diligence Assessment[/bold magenta]

💰 [bold]Valuation:[/bold] {assessment.overall_valuation}
📊 [bold]Financial Health:[/bold] [{financial_color}]{assessment.financial_health_score:.1%}[/{financial_color}]
🎯 [bold]Strategic Fit:[/bold] [{strategic_color}]{assessment.strategic_fit_score:.1%}[/{strategic_color}]

💪 [bold]Key Strengths:[/bold]
{chr(10).join(f'  • {strength}' for strength in assessment.key_strengths)}

⚠️  [bold]Critical Risks:[/bold]
{chr(10).join(f'  • {risk}' for risk in assessment.critical_risks)}

🔧 [bold]Integration Complexity:[/bold] {assessment.integration_complexity}
✨ [bold]Synergy Potential:[/bold] {assessment.synergy_potential}

🚨 [bold]Deal Breakers:[/bold]
{chr(10).join(f'  • {issue}' for issue in assessment.deal_breakers)}

📋 [bold]Final Recommendation:[/bold]
{assessment.recommendation}""",
        title="🔍 Due Diligence",
        border_style="magenta"
    ))


def display_transformation_plan(plan: DigitalTransformationPlan):
    """Display digital transformation plan results."""
    console.print(Panel.fit(
        f"""[bold cyan]Digital Transformation Strategy[/bold cyan]

📊 [bold]Current Maturity:[/bold] {plan.current_maturity_level}

🗺️  [bold]Transformation Roadmap:[/bold]
{chr(10).join(f'  • {step}' for step in plan.transformation_roadmap)}

⚙️  [bold]Technology Stack:[/bold]
{chr(10).join(f'  • {tech}' for tech in plan.technology_stack)}

🔄 [bold]Change Management:[/bold] {plan.change_management_approach}

💰 [bold]Investment Schedule:[/bold] {plan.investment_schedule}
📈 [bold]ROI Projections:[/bold] {plan.roi_projections}

⚠️  [bold]Implementation Risks:[/bold]
{chr(10).join(f'  • {risk}' for risk in plan.implementation_risks)}

🎯 [bold]Success Factors:[/bold]
{chr(10).join(f'  • {factor}' for factor in plan.success_factors)}""",
        title="💻 Digital Transformation",
        border_style="cyan"
    ))


def main():
    """Run comprehensive real-world kwargs demonstration."""
    console.print(Panel("🏢 Real-World Business Consulting Platform Demo", style="bold white on blue"))
    console.print("Demonstrating advanced kwargs usage in enterprise consulting scenarios\n")
    
    if not os.getenv("OPPER_API_KEY"):
        console.print("[bold red]Error:[/bold red] OPPER_API_KEY environment variable not set!")
        console.print("Please set your Opper API key to run the full demo:")
        console.print("export OPPER_API_KEY='your_api_key_here'")
        return
    
    try:
        # Create single provider and unified trace for all scenarios
        provider = OpperProvider()
        console.print("🚀 [bold]Starting unified business consulting workflow...[/bold]")
        trace_id = provider.start_trace("business_consulting_platform", "Comprehensive enterprise consulting across 4 domains")
        console.print(f"✅ Started unified trace: [cyan]{trace_id}[/cyan]")
        
        # Run all scenarios under the same trace
        scenarios = [
            ("Market Entry Analysis", scenario_market_entry),
            ("Product Launch Strategy", scenario_product_launch),
            ("M&A Due Diligence", scenario_due_diligence),
            ("Digital Transformation", scenario_digital_transformation)
        ]
        
        results = {}
        for name, scenario_func in scenarios:
            console.print(f"\n{'='*80}\n")
            console.print(f"🔄 [bold]Running {name} (trace: {provider.current_trace_id})...[/bold]")
            results[name] = scenario_func(provider)  # Pass provider to each scenario
        
        # Summary
        console.print(f"\n{'='*80}\n")
        console.print(Panel("🎉 All Consulting Scenarios Completed!", style="bold green"))
        
        # Create summary table
        summary_table = Table(title="Consultation Results Summary")
        summary_table.add_column("Scenario", style="cyan", no_wrap=True)
        summary_table.add_column("Key Metric", style="yellow")
        summary_table.add_column("Recommendation", style="green")
        
        if results.get("Market Entry Analysis"):
            analysis = results["Market Entry Analysis"]
            summary_table.add_row(
                "Market Entry",
                f"{analysis.success_probability:.1%} success rate",
                "Proceed with pilot" if analysis.success_probability > 0.7 else "Requires refinement"
            )
        
        if results.get("Product Launch Strategy"):
            strategy = results["Product Launch Strategy"]
            summary_table.add_row(
                "Product Launch",
                f"Multi-channel approach",
                "Execute launch strategy"
            )
        
        if results.get("M&A Due Diligence"):
            assessment = results["M&A Due Diligence"]
            summary_table.add_row(
                "M&A Assessment",
                f"{assessment.strategic_fit_score:.1%} strategic fit",
                "Proceed" if assessment.strategic_fit_score > 0.7 else "Negotiate terms"
            )
        
        if results.get("Digital Transformation"):
            plan = results["Digital Transformation"]
            summary_table.add_row(
                "Digital Transform",
                f"Phased approach",
                "Begin Phase 1 implementation"
            )
        
        console.print(summary_table)
        
        # End the unified trace
        final_summary = f"Completed comprehensive business consulting across 4 domains: {list(results.keys())}"
        provider.end_trace(final_summary)
        console.print(f"✅ [bold]Unified trace completed:[/bold] [cyan]{trace_id}[/cyan]")
        
        console.print("\n[bold cyan]Advanced Features Demonstrated:[/bold cyan]")
        console.print("• 🎯 [bold]Rich Context Data:[/bold] Complex business scenarios with detailed kwargs")
        console.print("• 📊 [bold]Structured Analysis:[/bold] Domain-specific output schemas")
        console.print("• 🔧 [bold]Flexible Input:[/bold] Dict-based structured input to Opper")
        console.print("• 📈 [bold]Enterprise Complexity:[/bold] Real-world business consulting scenarios")
        console.print("• 🚀 [bold]Scalable Patterns:[/bold] Reusable consulting methodology")
        console.print("• 🔗 [bold]Unified Tracing:[/bold] All 4 scenarios grouped under one trace")
        
        console.print(f"\n[bold yellow]💡 Check Opper dashboard for unified trace: {trace_id}[/bold yellow]")
        console.print("All consulting scenarios should appear as child spans under the main trace!")
        
    except Exception as e:
        console.print(f"[bold red]Error running demo:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
