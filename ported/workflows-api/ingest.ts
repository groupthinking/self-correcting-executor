import { type NextRequest, NextResponse } from "next/server"
import { createServerClient } from "@/lib/supabase/server"
import { fetchUrlContent, isVideoUrl, isPdfUrl, isDocumentUrl, extractYouTubeVideoId } from "@/lib/services/fetcher"
import { transcribeVideo, transcribeYouTubeVideo } from "@/lib/services/transcriber"
import { extractPdfText } from "@/lib/services/pdf-extractor"
import { extractGoogleDocText } from "@/lib/services/doc-extractor"
import type { SourceType } from "@/lib/types/workflow"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { url, language } = body

    if (!url || typeof url !== "string") {
      return NextResponse.json({ error: "URL is required" }, { status: 400 })
    }

    const supabase = await createServerClient()
    const {
      data: { user },
    } = await supabase.auth.getUser()

    // Determine source type
    let sourceType: SourceType
    let content: string
    let metadata: Record<string, unknown> = {}

    if (isVideoUrl(url)) {
      sourceType = "video"
      console.log("[v0] Processing video URL:", url)

      // Check if YouTube
      const youtubeId = extractYouTubeVideoId(url)
      if (youtubeId) {
        const result = await transcribeYouTubeVideo(youtubeId, language, user?.id)
        content = result.text
        metadata = { ...result.metadata, videoId: youtubeId }
      } else {
        const result = await transcribeVideo(url, language)
        content = result.text
        metadata = result.metadata || {}
      }
    } else if (isPdfUrl(url)) {
      sourceType = "pdf"
      console.log("[v0] Processing PDF URL:", url)
      const result = await extractPdfText(url)
      content = result.text
      metadata = result.metadata || {}
    } else if (isDocumentUrl(url)) {
      sourceType = "doc"
      console.log("[v0] Processing document URL:", url)
      const result = await extractGoogleDocText(url, user?.id)
      content = result.text
      metadata = result.metadata || {}
    } else {
      sourceType = "url"
      console.log("[v0] Processing web URL:", url)
      const result = await fetchUrlContent(url)
      content = result.content
      metadata = result.metadata
    }

    // Create workflow run in database
    const { data: workflowRun, error: workflowError } = await supabase
      .from("workflow_runs")
      .insert({
        status: "ingesting",
        source_type: sourceType,
        source_url: url,
        metadata: {
          language,
          contentLength: content.length,
          ...metadata,
        },
      })
      .select()
      .single()

    if (workflowError) {
      console.error("[v0] Failed to create workflow run:", workflowError)
      return NextResponse.json(
        { error: "Failed to create workflow run", details: workflowError.message },
        { status: 500 },
      )
    }

    console.log("[v0] Created workflow run:", workflowRun.id)

    return NextResponse.json({
      success: true,
      workflowRunId: workflowRun.id,
      sourceType,
      contentLength: content.length,
      content, // Return full content instead of preview
      metadata,
    })
  } catch (error) {
    console.error("[v0] Ingestion error:", error)
    return NextResponse.json(
      {
        error: "Failed to ingest content",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
