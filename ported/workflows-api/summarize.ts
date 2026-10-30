import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { extractExecSummary } from "@/lib/services/summarizer"
import type { AIProvider } from "@/lib/services/ai-provider"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { workflowRunId, content, provider = "openai" } = body

    if (!workflowRunId || !content) {
      return NextResponse.json({ error: "workflowRunId and content are required" }, { status: 400 })
    }

    console.log("[v0] Summarizing content for workflow run:", workflowRunId)

    // Extract executive summary using AI
    const summary = await extractExecSummary(content, provider as AIProvider)

    // Store in database
    const supabase = await createClient()

    const { data: execSummary, error: summaryError } = await supabase
      .from("exec_summaries")
      .insert({
        workflow_run_id: workflowRunId,
        title: summary.title,
        goal: summary.goal,
        key_requirements: summary.key_requirements,
        constraints: summary.constraints,
        assumptions: summary.assumptions,
        success_metrics: summary.success_metrics,
        raw_content: content,
        metadata: {
          provider,
          extractedAt: new Date().toISOString(),
        },
      })
      .select()
      .single()

    if (summaryError) {
      console.error("[v0] Failed to store executive summary:", summaryError)
      return NextResponse.json(
        { error: "Failed to store executive summary", details: summaryError.message },
        { status: 500 },
      )
    }

    // Update workflow run status
    await supabase.from("workflow_runs").update({ status: "planning" }).eq("id", workflowRunId)

    console.log("[v0] Executive summary created:", execSummary.id)

    return NextResponse.json({
      success: true,
      execSummary,
    })
  } catch (error) {
    console.error("[v0] Summarization error:", error)
    return NextResponse.json(
      {
        error: "Failed to generate executive summary",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
