import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { generateCodeScaffold } from "@/lib/services/code-generator"
import type { BuildPlan } from "@/lib/types/workflow"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { buildPlanId } = body

    if (!buildPlanId) {
      return NextResponse.json({ error: "buildPlanId is required" }, { status: 400 })
    }

    console.log("[v0] Generating code scaffold for build plan:", buildPlanId)

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

    // Generate code scaffold
    const scaffold = await generateCodeScaffold(buildPlan as unknown as BuildPlan)

    // Store deployment artifacts
    for (const file of scaffold.files) {
      await supabase.from("deployment_artifacts").insert({
        workflow_run_id: buildPlan.workflow_run_id,
        build_plan_id: buildPlanId,
        artifact_type: "code_scaffold",
        file_path: file.path,
        content: file.content,
        metadata: {
          language: file.language,
          generatedAt: new Date().toISOString(),
        },
      })
    }

    // Store README as separate artifact
    await supabase.from("deployment_artifacts").insert({
      workflow_run_id: buildPlan.workflow_run_id,
      build_plan_id: buildPlanId,
      artifact_type: "code_scaffold",
      file_path: "README.md",
      content: scaffold.readme,
      metadata: {
        language: "markdown",
        generatedAt: new Date().toISOString(),
      },
    })

    console.log("[v0] Code scaffold generated successfully")

    return NextResponse.json({
      success: true,
      scaffold: {
        fileCount: scaffold.files.length,
        structure: scaffold.structure,
        files: scaffold.files.map((f) => ({
          path: f.path,
          language: f.language,
          size: f.content.length,
        })),
      },
    })
  } catch (error) {
    console.error("[v0] Scaffold generation error:", error)
    return NextResponse.json(
      {
        error: "Failed to generate code scaffold",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
