import json
from pathlib import Path
import shutil
import subprocess

from protocols.file_validator import task as validate
from workflows.plan_scaffold import generate_plan_scaffold
from test_build_plan import plan


def test_plan_scaffold_compiles_and_node_executes_real_dependencies_and_correction(
    tmp_path,
):
    value = plan(
        tasks=[
            {"id": "last", "name": "Double", "depends_on": ["first"]},
            {"id": "first", "name": "Parse", "depends_on": []},
        ]
    )
    artifact = generate_plan_scaffold(value)
    assert validate(artifact)["success"], validate(artifact)
    for file in artifact["files"]:
        destination = tmp_path / file["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(file["content"])
    assert (
        json.loads((tmp_path / "package.json").read_text())["devDependencies"][
            "typescript"
        ]
        == "5.9.3"
    )
    compiler = (
        Path(__file__).resolve().parents[1]
        / "tools/typescript-validation/node_modules/typescript/bin/tsc"
    )
    node = shutil.which("node")
    subprocess.run(
        [node, str(compiler), "-p", "tsconfig.json"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    script = """import {execute,plan} from './dist/index.js';
const events=[];
const result = await execute({handlers:{worker:async(input,ctx)=> {
 if(ctx.task.id==='first') return JSON.parse(input.source);
 return {value:ctx.dependencies.first.value*2};
}},repairs:{first: async(error,input,attempt)=>({input:{...input,source:input.approved},before:input.source,after:input.approved,reason:'Restore supplied source'})},onEvent:e=>events.push(e)}, {source:'broken',approved:'{"value":21}'});
if(result.status!=='completed'||result.outputs.last.value!==42) throw new Error(JSON.stringify(result));
if(events.filter(e=>e.type==='trajectory'&&e.status==='failed').length!==1) throw new Error('Lost failure');
const missing=await execute({handlers:{}},{});
if(missing.status!=='failed') throw new Error('Missing handler falsely succeeded');
const {execute:executeGraph}=await import('./dist/runtime.js');
const kinds=['SequentialAgent','ParallelAgent','LoopAgent','LlmAgent','CustomAgent'];
const typedPlan={...plan,agents:kinds.map(kind=>({name:kind,kind,description:kind})),tasks:kinds.map((kind,i)=>({id:String(i),name:kind,depends_on:[]}))};
const typed=await executeGraph(typedPlan,{handlers:{CustomAgent:async input=>({value:input.value+1}),LlmAgent:async input=>({value:input.value*2})},composites:{SequentialAgent:{children:['CustomAgent','LlmAgent']},ParallelAgent:{children:['CustomAgent','LlmAgent']},LoopAgent:{children:['CustomAgent'],maxIterations:3,until:value=>value.value===3}}},{value:1});
if(typed.status!=='completed'||typed.outputs['0'].value!==4||typed.outputs['1'].CustomAgent.value!==2||typed.outputs['2'].value!==3||typed.outputs['3'].value!==2) throw new Error(JSON.stringify(typed));
console.log(JSON.stringify({result,missing:missing.status,planId:plan.id,typed:typed.status}));
"""
    (tmp_path / "verify.mjs").write_text(script)
    executed = subprocess.run(
        [node, "verify.mjs"], cwd=tmp_path, text=True, capture_output=True, check=True
    )
    assert json.loads(executed.stdout)["result"]["outputs"]["last"] == {"value": 42}


def test_documented_demo_executes_actual_graph_and_persisted_artifact_readback(
    tmp_path,
):
    import sys
    from workflows.build_plan import WorkflowStore

    database = tmp_path / "demo.sqlite3"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.build_plan_workflow",
            "--demo",
            "--database",
            str(database),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    # Native optional-agent imports can write startup diagnostics; parse the JSON body.
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["execution"]["outputs"]["double"] == {"value": 20}
    assert (
        payload["artifacts_saved"] >= 8
        and payload["deployment_status"] == "scaffold_only"
    )
    store = WorkflowStore(database)
    assert len(store.all("artifacts")) == payload["artifacts_saved"]
    assert store.get("runs", "local-example-run")["status"] == "completed"
    store.close()


def test_generated_runtime_partial_effect_observation_and_protocol_regressions(
    tmp_path,
):
    artifact = generate_plan_scaffold(plan())
    for file in artifact["files"]:
        destination = tmp_path / file["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(file["content"])
    node = shutil.which("node")
    compiler = (
        Path(__file__).resolve().parents[1]
        / "tools/typescript-validation/node_modules/typescript/bin/tsc"
    )
    subprocess.run(
        [node, str(compiler), "-p", "tsconfig.json"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    script = r"""import assert from 'node:assert/strict';
import {execute} from './dist/runtime.js';
import {plan} from './dist/plan.js';
let effects=0;
const observed=await execute(plan,{handlers:{worker:async()=>{effects++;return {receipt:'committed'};}},onEvent:event=>{if(event.type==='trajectory'&&event.status==='completed')throw Error('store down');}},{});
assert.equal(observed.status,'failed');assert.equal(effects,1);
assert.equal(observed.events.filter(e=>e.type==='trajectory').length,1);
for(const kind of ['SequentialAgent','ParallelAgent','LoopAgent']) {
 const hits=[],checks=[],order=[];
 const source={...plan,agents:[{name:'root',kind,description:''},{name:'effect',kind:'CustomAgent',description:''},{name:'check',kind:'CustomAgent',description:''}]};
 const result=await execute(source,{handlers:{effect:async input=>{if(kind==='ParallelAgent')await new Promise(r=>setTimeout(r,20));hits.push(input.value+1);order.push('effect');return {value:input.value+1};},check:async input=>{checks.push(input.value);order.push('check');const target=kind==='LoopAgent'?2:(kind==='ParallelAgent'?0:1);if(input.value===target&&checks.filter(v=>v===target).length===1)throw Error('failed once');return input;}},composites:{root:{children:['effect','check'],maxIterations:3,until:v=>v.value===2}}},{value:0});
 assert.equal(result.status,'completed',JSON.stringify(result));assert.deepEqual(hits,kind==='LoopAgent'?[1,2]:[1]);
 if(kind==='ParallelAgent'){assert.deepEqual(order,['check','effect','check']);await new Promise(r=>setTimeout(r,30));assert.deepEqual(hits,[1]);}
}
const mcpResult=await execute({...plan,mcp_tools:['failure']},{handlers:{worker:async(i,c)=>c.callTool('mcp','failure',{})},protocols:{mcp:{listTools:async()=>['failure'],callTool:async()=>({isError:true,content:[{text:'denied'}]})}}},{});
assert.equal(mcpResult.status,'failed');assert.equal(mcpResult.events.filter(e=>e.type==='tool_call'&&e.status==='completed').length,0);
let paid=0;
for(const cap of [NaN,Infinity,0,-1]) {
 const payment=await execute({...plan,ap2_required:true},{handlers:{worker:async(i,c)=>c.callTool('ap2','vendor',{amount:2,currency:'USD',idempotencyKey:'approved-1'})},protocols:{mandate:{maxAmount:cap,vendorId:'vendor',currency:'USD'},ap2:{validateMandate:async()=>true,pay:async()=>{paid++;return {status:'authorized'};}}}},{});
 assert.equal(payment.status,'failed');
}
assert.equal(paid,0);
console.log(JSON.stringify({observation:'no replay',composites:'checkpointed',mcp:'error retained',ap2:'no payment'}));
"""
    (tmp_path / "regressions.mjs").write_text(script)
    result = subprocess.run(
        [node, "regressions.mjs"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["ap2"] == "no payment"
