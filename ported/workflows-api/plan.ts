import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { generateBuildPlan } from "@/lib/services/planner"
import type { AIProvider } from "@/lib/services/ai-provider"
import type { ExecSummaryOutput } from "@/lib/services/summarizer"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { workflowRunId, execSummaryId, provider = "openai" } = body

    if (!workflowRunId || !execSummaryId) {
      return NextResponse.json({ error: "workflowRunId and execSummaryId are required" }, { status: 400 })
    }

    console.log("[v0] Generating build plan for workflow run:", workflowRunId)

    // Get executive summary from database
    const supabase = await createClient()

    const { data: execSummary, error: summaryError } = await supabase
      .from("exec_summaries")
      .select("*")
      .eq("id", execSummaryId)
      .single()

    if (summaryError || !execSummary) {
      return NextResponse.json({ error: "Executive summary not found" }, { status: 404 })
    }

    // Convert to ExecSummaryOutput format
    const summaryInput: ExecSummaryOutput = {
      title: execSummary.title,
      goal: execSummary.goal,
      key_requirements: execSummary.key_requirements as string[],
      constraints: execSummary.constraints as string[],
      assumptions: execSummary.assumptions as string[],
      success_metrics: execSummary.success_metrics as string[],
    }

    // Generate build plan using AI
    const plan = await generateBuildPlan(summaryInput, provider as AIProvider)

    // Store in database
    const { data: buildPlan, error: planError } = await supabase
      .from("build_plans")
      .insert({
        workflow_run_id: workflowRunId,
        exec_summary_id: execSummaryId,
        summary_title: plan.summary_title,
        summary_goal: plan.summary_goal,
        agents: plan.agents,
        tasks: plan.tasks,
        mcp_tools: plan.mcp_tools,
        a2a_required: plan.a2a_required,
        ap2_required: plan.ap2_required,
        metadata: {
          provider,
          generatedAt: new Date().toISOString(),
        },
      })
      .select()
      .single()

    if (planError) {
      console.error("[v0] Failed to store build plan:", planError)
      return NextResponse.json({ error: "Failed to store build plan", details: planError.message }, { status: 500 })
    }

    // Update workflow run status
    await supabase.from("workflow_runs").update({ status: "compiling" }).eq("id", workflowRunId)

    console.log("[v0] Build plan created:", buildPlan.id)

    return NextResponse.json({
      success: true,
      buildPlan,
    })
  } catch (error) {
    console.error("[v0] Planning error:", error)
    return NextResponse.json(
      {
        error: "Failed to generate build plan",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
