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

    console.log("[v0] Compiling workflow for build plan:", buildPlanId)

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

    // Compile workflow
    const compiler = new WorkflowCompiler()
    const compiledWorkflow = await compiler.compile(buildPlan as unknown as BuildPlan)

    // Store agent trajectories
    for (const agent of compiledWorkflow.agentGraph) {
      await supabase.from("agent_trajectories").insert({
        workflow_run_id: buildPlan.workflow_run_id,
        build_plan_id: buildPlanId,
        agent_name: agent.name,
        agent_kind: agent.kind,
        step_number: 0,
        action: "initialized",
        status: "completed",
        metadata: {
          tools: agent.tools,
          description: agent.description,
        },
      })
    }

    // Update workflow run status
    await supabase.from("workflow_runs").update({ status: "deploying" }).eq("id", buildPlan.workflow_run_id)

    console.log("[v0] Workflow compiled successfully")

    return NextResponse.json({
      success: true,
      workflow: {
        workflowId: compiledWorkflow.workflowId,
        agentCount: compiledWorkflow.agentGraph.length,
        taskCount: compiledWorkflow.taskGraph.length,
        agents: compiledWorkflow.agentGraph,
        tasks: compiledWorkflow.taskGraph,
        metadata: compiledWorkflow.metadata,
      },
    })
  } catch (error) {
    console.error("[v0] Compilation error:", error)
    return NextResponse.json(
      {
        error: "Failed to compile workflow",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
