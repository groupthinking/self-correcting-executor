// TypeScript types for workflow generation

export type SourceType = "url" | "video" | "pdf" | "doc"

export type WorkflowStatus = "pending" | "ingesting" | "planning" | "compiling" | "deploying" | "completed" | "failed"

export type AgentKind = "SequentialAgent" | "ParallelAgent" | "LoopAgent" | "LlmAgent" | "CustomAgent"

export type Protocol = "mcp" | "a2a" | "ap2"

export type ArtifactType = "code_scaffold" | "full_app" | "docker_image" | "deployment_config"

export interface WorkflowRun {
  id: string
  created_at: string
  updated_at: string
  status: WorkflowStatus
  source_type: SourceType
  source_url: string
  error_message?: string
  metadata?: Record<string, unknown>
}

export interface ExecSummary {
  id: string
  workflow_run_id: string
  created_at: string
  title: string
  goal: string
  key_requirements: string[]
  constraints: string[]
  assumptions: string[]
  success_metrics: string[]
  raw_content?: string
  metadata?: Record<string, unknown>
}

export interface AgentSpec {
  name: string
  kind: AgentKind
  description: string
}

export interface Task {
  id: string
  name: string
  depends_on: string[]
}

export interface BuildPlan {
  id: string
  workflow_run_id: string
  exec_summary_id: string
  created_at: string
  summary_title: string
  summary_goal: string
  agents: AgentSpec[]
  tasks: Task[]
  mcp_tools: string[]
  a2a_required: boolean
  ap2_required: boolean
  metadata?: Record<string, unknown>
}

export interface AgentTrajectory {
  id: string
  workflow_run_id: string
  build_plan_id?: string
  created_at: string
  agent_name: string
  agent_kind: AgentKind
  step_number: number
  action: string
  input?: Record<string, unknown>
  output?: Record<string, unknown>
  duration_ms?: number
  status: "started" | "completed" | "failed"
  error_message?: string
  metadata?: Record<string, unknown>
}

export interface ToolCall {
  id: string
  trajectory_id: string
  created_at: string
  protocol: Protocol
  tool_name: string
  parameters?: Record<string, unknown>
  result?: Record<string, unknown>
  duration_ms?: number
  status: "started" | "completed" | "failed"
  error_message?: string
  metadata?: Record<string, unknown>
}

export interface DeploymentArtifact {
  id: string
  workflow_run_id: string
  build_plan_id: string
  created_at: string
  artifact_type: ArtifactType
  file_path?: string
  content?: string
  storage_url?: string
  metadata?: Record<string, unknown>
}

export interface CodeFile {
  path: string
  content: string
  language: string
}

export interface CodeScaffold {
  files: CodeFile[]
  structure: string
  readme: string
}
