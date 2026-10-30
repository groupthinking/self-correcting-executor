// Executable adaptation of archived workflows/lib/services/workflow-compiler.ts.
// No simulated tasks, inferred child membership, or demo payment mandates.
import type { BuildPlan, AgentSpec, Task } from './types.js';
export type Context = {
  task: Task; dependencies: Record<string, unknown>; plan: BuildPlan;
  callTool(protocol: 'mcp'|'a2a'|'ap2', name: string, parameters: Record<string, unknown>): Promise<unknown>;
};
export type Handler = (input: unknown, context: Context) => Promise<unknown> | unknown;
export type Repair = { input: unknown; before: unknown; after: unknown; reason: string };
export type Event = Record<string, unknown>;
export type Bindings = {
  handlers: Record<string, Handler>;
  composites?: Record<string, {children: string[]; maxIterations?: number; until?: (value: unknown, iteration: number) => boolean}>;
  repairs?: Record<string, (error: unknown, input: unknown, attempt: number) => Promise<Repair> | Repair>;
  attempts?: number; selfCorrection?: boolean; onEvent?: (event: Event) => Promise<void> | void;
  protocols?: {
    mcp?: {listTools(): Promise<string[]>; callTool(name: string, args: Record<string, unknown>): Promise<unknown>};
    a2a?: {discoverAgents(capability: string): Promise<unknown[]>; dispatchTask(name: string, args: Record<string, unknown>): Promise<unknown>};
    ap2?: {validateMandate(mandate: Record<string, unknown>): Promise<boolean>; pay(mandate: Record<string, unknown>, args: Record<string, unknown>): Promise<unknown>};
    mandate?: Record<string, unknown>;
  };
};
class NonRetryableError extends Error { readonly retryable = false; }
const nonRetryable = (value: unknown): boolean => typeof value === 'object' && value !== null && 'retryable' in value && value.retryable === false;
const failed = (value: unknown): boolean => typeof value === 'object' && value !== null &&
  (('isError' in value && value.isError === true) || ('success' in value && value.success === false) || ('status' in value && ['failed','error'].includes(String(value.status))));
export function compile(plan: BuildPlan) {
  const copy = structuredClone(plan);
  if (!copy.agents.length || !copy.tasks.length) throw new Error('Plan requires agents and tasks');
  const ids = new Set(copy.tasks.map(t => t.id));
  const names = new Set(copy.agents.map(a => a.name));
  const kinds = ['SequentialAgent','ParallelAgent','LoopAgent','LlmAgent','CustomAgent'];
  if (ids.size !== copy.tasks.length || names.size !== copy.agents.length) throw new Error('Duplicate names or IDs');
  for (const agent of copy.agents) if (!kinds.includes(agent.kind)) throw new Error('Unknown agent kind');
  const pending = new Map(copy.tasks.map(t => [t.id, t.depends_on]));
  const done = new Set<string>();
  while (pending.size) {
    const ready = [...pending].filter(([, deps]) => deps.every(d => done.has(d)));
    if (!ready.length) throw new Error('Cyclic or missing task dependencies');
    for (const [id] of ready) { done.add(id); pending.delete(id); }
  }
  return { plan: copy, agentGraph: copy.agents.map((a,i) => ({id: `agent_${i}`, ...a, tools: [...copy.mcp_tools]})),
    taskGraph: copy.tasks.map((t,i) => ({...t, agentId: `agent_${i % copy.agents.length}`, status: 'pending'})) };
}
export async function execute(source: BuildPlan, bindings: Bindings, input: unknown = {}) {
  const graph = compile(source), plan = graph.plan;
  const outputs: Record<string, unknown> = Object.create(null), events: Event[] = [];
  const emit = async (event: Event) => { events.push(structuredClone(event)); try { await bindings.onEvent?.(structuredClone(event)); } catch(error) { throw new NonRetryableError(`Observation failed; no replay: ${String(error)}`); } };
  const protocols = bindings.protocols ?? {};
  const tool = async (protocol: 'mcp'|'a2a'|'ap2', name: string, args: Record<string, unknown>) => {
    await emit({type: 'tool_call', status: 'started', protocol, name, parameters: args});
    let result: unknown;
    try {
      if (protocol === 'mcp' && plan.mcp_tools.includes(name) && protocols.mcp) result = await protocols.mcp.callTool(name, args);
      else if (protocol === 'a2a' && plan.a2a_required && protocols.a2a) result = await protocols.a2a.dispatchTask(name, args);
      else if (protocol === 'ap2' && plan.ap2_required && protocols.ap2 && protocols.mandate) {
        const mandate = protocols.mandate;
        if (!(await protocols.ap2.validateMandate(mandate)) || typeof args.amount !== 'number' || !Number.isFinite(args.amount) ||
          args.amount <= 0 || typeof mandate.maxAmount !== 'number' || !Number.isFinite(mandate.maxAmount) || mandate.maxAmount <= 0 || args.amount > mandate.maxAmount ||
          args.currency !== mandate.currency || name !== mandate.vendorId || typeof args.idempotencyKey !== 'string' || !args.idempotencyKey)
          throw new Error('Payment does not match caller authorization and idempotency key');
        result = await protocols.ap2.pay(mandate, args);
      } else throw new Error('Protocol call is undeclared or unconfigured');
      if (failed(result)) throw new Error('Protocol reported failure');
      if ((protocol === 'a2a' || protocol === 'ap2') && (typeof result !== 'object' || result === null || !('status' in result) ||
          !(protocol === 'ap2' ? ['authorized','completed'] : ['completed']).includes(String(result.status))))
        throw new Error('Protocol outcome remains incomplete');
      await emit({type: 'tool_call', status: 'completed', protocol, name, result});
      return result;
    } catch(error) { if (nonRetryable(error)) throw error; await emit({type: 'tool_call', status: 'failed', protocol, name, error: String(error)}); throw error; }
  };
  type Checkpoints = Map<string,{input:unknown;output:unknown}>;
  const agent = async (spec: AgentSpec, value: unknown, context: Context, checkpoints: Checkpoints, path: string[] = [], stack: string[] = []): Promise<unknown> => {
    const originalInput=structuredClone(value);
    const identity = [...path,spec.name], key=JSON.stringify(identity), saved=checkpoints.get(key);
    if(saved) {
      if(JSON.stringify(saved.input)!==JSON.stringify(value)) throw new NonRetryableError('Retry input changed after completed child; reconcile before replay');
      return structuredClone(saved.output);
    }
    if (stack.includes(spec.name)) throw new Error('Composite cycle');
    let result: unknown;
    if (spec.kind === 'LlmAgent' || spec.kind === 'CustomAgent') {
      const handler = bindings.handlers[spec.name];
      if (!handler) throw new Error(`Missing executable binding: ${spec.name}`);
      result = await handler(structuredClone(value), context);
      if (failed(result)) throw new Error(`Agent ${spec.name} reported failure`);
    } else {
      const config = bindings.composites?.[spec.name];
      if (!config?.children.length) throw new Error(`Missing composite controls: ${spec.name}`);
      const children = config.children.map(name => { const child = plan.agents.find(a => a.name === name); if (!child) throw new Error('Missing child'); return child; });
      const run = (child: AgentSpec, data: unknown, index:number, iteration=0) => agent(child, structuredClone(data), context, checkpoints, [...identity,String(iteration),String(index)], [...stack,spec.name]);
      if (spec.kind === 'ParallelAgent') {
        if(new Set(config.children).size!==config.children.length) throw new Error('Parallel children must have unique names');
        const settled=await Promise.allSettled(children.map((c,i)=>run(c,value,i)));
        const errors=settled.filter((v):v is PromiseRejectedResult=>v.status==='rejected');
        if(errors.length) throw (errors.find(v=>nonRetryable(v.reason))??errors[0]).reason;
        result=Object.fromEntries(settled.map((v,i)=>[children[i].name,(v as PromiseFulfilledResult<unknown>).value]));
      } else {
        const sequence=async(data:unknown,iteration=0)=>{ for(let i=0;i<children.length;i++) data=await run(children[i],data,i,iteration); return data; };
        if(spec.kind==='SequentialAgent') result=await sequence(value);
        else {
          const limit=config.maxIterations;
          if(!Number.isInteger(limit)||!limit||limit<1||limit>1000||!config.until) throw new Error('Loop requires explicit bound and termination condition');
          let stopped=false;
          for(let i=1;i<=limit;i++) { value=await sequence(value,i); if(config.until(value,i) === true) { stopped=true;break; } }
          if(!stopped) throw new Error('Loop did not reach stop condition');
          result=value;
        }
      }
    }
    try { checkpoints.set(key,{input:structuredClone(originalInput),output:structuredClone(result)}); }
    catch(error) { throw new NonRetryableError('Completed child could not be checkpointed; no replay'); }
    return result;
  };
  try {
    await emit({type: 'workflow_status', status: 'compiling'});
    if (plan.mcp_tools.length) {
      if (!protocols.mcp) throw new Error('MCP client required');
      const tools = await protocols.mcp.listTools();
      if (!plan.mcp_tools.every(t => tools.includes(t))) throw new Error('MCP tool missing');
    }
    if (plan.a2a_required && (!protocols.a2a || !(await protocols.a2a.discoverAgents('code_generation')).length)) throw new Error('A2A registry required');
    if (plan.ap2_required && (!protocols.ap2 || !protocols.mandate || !(await protocols.ap2.validateMandate(protocols.mandate)))) throw new Error('Authorized AP2 mandate required');
    const count = bindings.selfCorrection === false ? 1 : (bindings.attempts ?? 2);
    if (!Number.isInteger(count) || count < 1 || count > 20) throw new Error('Attempts must be bounded');
    const pending = [...graph.taskGraph];
    while (pending.length) {
      const ready = pending.filter(t => t.depends_on.every(d => Object.hasOwn(outputs,d)));
      if (!ready.length) throw new Error('Graph made no progress');
      for (const task of ready) {
        const spec = graph.agentGraph.find(a => a.id === task.agentId)!;
        const context = {task, plan, dependencies: Object.fromEntries(task.depends_on.map(d => [d,structuredClone(outputs[d])])), callTool: tool};
        const checkpoints:Checkpoints=new Map();
        let current = structuredClone(input), completed = false;
        task.status = 'running';
        for (let attempt=1; attempt<=count; attempt++) {
          try {
            const output = await agent(spec,current,context,checkpoints);
            if (failed(output)) throw new Error('Task reported failure');
            outputs[task.id] = structuredClone(output); task.status = 'completed'; completed = true;
            await emit({type:'trajectory',taskId:task.id,agentKind:spec.kind,status:'completed',attempt,output}); break;
          } catch(error) {
            if(nonRetryable(error)) throw error;
            await emit({type:'trajectory',taskId:task.id,agentKind:spec.kind,status:'failed',attempt,error:String(error)});
            const repair = bindings.repairs?.[task.id];
            if (attempt < count && repair) {
              const change = await repair(error,structuredClone(current),attempt);
              if (!change.reason || JSON.stringify(change.before) === JSON.stringify(change.after)) throw new Error('Repair made no concrete change');
              current = structuredClone(change.input);
              await emit({type:'correction',taskId:task.id,attempt,before:change.before,after:change.after,reason:change.reason});
            }
            if (attempt === count) { task.status='failed'; throw error; }
          }
        }
        if (!completed) throw new Error('Task incomplete');
        pending.splice(pending.indexOf(task),1);
      }
    }
    await emit({type:'workflow_status',status:'completed'});
    return {status:'completed',outputs,events,taskGraph:graph.taskGraph};
  } catch(error) {
    for (const task of graph.taskGraph) if (task.status === 'running') task.status = 'failed';
    try { await emit({type:'workflow_status',status:'failed',error:String(error)}); } catch { /* Return failed with captured in-memory events; never replay work. */ }
    return {status:'failed',error:String(error),outputs,events,taskGraph:graph.taskGraph};
  }
}
