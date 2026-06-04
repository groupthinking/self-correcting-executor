import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { WorkflowCompiler } from "@/lib/services/workflow-compiler"
import type { BuildPlan } from "@/lib/types/workflow"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { buildPlanId } = body

    if (!buildPlanId) {
      return NextResponse.json({ error: "buildPlanId is required" }, { status: 400 })
    }

    console.log("[v0] Executing workflow for build plan:", buildPlanId)

    // Get build plan from database
    const supabase = await createClient()

    const { data: buildPlan, error: planError } = await supabase
      .from("build_plans")
      .select("*")
      .eq("id", buildPlanId)
      .single()

    if (planError || !buildPlan) {
      return NextResponse.json({ error: "Build plan not found" }, { status: 404 })
    }

    // Compile and execute workflow
    const compiler = new WorkflowCompiler()
    const compiledWorkflow = await compiler.compile(buildPlan as unknown as BuildPlan)
    await compiler.execute(compiledWorkflow)

    // Update workflow run status
    await supabase.from("workflow_runs").update({ status: "completed" }).eq("id", buildPlan.workflow_run_id)

    console.log("[v0] Workflow executed successfully")

    return NextResponse.json({
      success: true,
      message: "Workflow executed successfully",
      workflowId: compiledWorkflow.workflowId,
    })
  } catch (error) {
    console.error("[v0] Execution error:", error)
    return NextResponse.json(
      {
        error: "Failed to execute workflow",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
