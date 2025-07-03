# Code Generation Agent
# Specialized agent for generating API endpoint code

import textwrap
from datetime import datetime
from typing import Dict, Any


class CodeGeneratorAgent:
    """Agent specialized in generating code based on intent"""

    def __init__(self):
        self.name = "code_generator"
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load code generation templates"""
        return {
            "fastapi_endpoint": textwrap.dedent(
                """
                @app.post("/api/v1/{endpoint_name}")
                async def {function_name}({parameters}):
                    \"\"\"
                    {description}
                    \"\"\"
                    try:
                        # Validate input
                        {validation_logic}

                        # Process request
                        {processing_logic}

                        # Return response
                        return {{
                            "status": "success",
                            "result": result,
                            "timestamp": datetime.utcnow().isoformat()
                        }}
                    except ValidationError as e:
                        raise HTTPException(status_code=400, detail=str(e))
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=str(e))
            """
            ),
            "rest_api": textwrap.dedent(
                """
                # {title}
                # Generated API endpoint

                from fastapi import FastAPI, HTTPException
                from pydantic import BaseModel
                from datetime import datetime
                from typing import Optional, List

                {models}

                {endpoints}
            """
            ),
            "crud_operations": textwrap.dedent(
                """
                # CRUD operations for {entity}

                @app.post("/{entity_plural}")
                async def create_{entity}(item: {model_name}):
                    \"\"\"Create new {entity}\"\"\"
                    # Implementation here
                    pass

                @app.get("/{entity_plural}/{{id}}")
                async def get_{entity}(id: int):
                    \"\"\"Get {entity} by ID\"\"\"
                    # Implementation here
                    pass

                @app.put("/{entity_plural}/{{id}}")
                async def update_{entity}(id: int, item: {model_name}):
                    \"\"\"Update {entity}\"\"\"
                    # Implementation here
                    pass

                @app.delete("/{entity_plural}/{{id}}")
                async def delete_{entity}(id: int):
                    \"\"\"Delete {entity}\"\"\"
                    # Implementation here
                    pass
            """
            ),
        }

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation based on inputs"""
        intent = inputs.get("intent", "")
        context = inputs.get("context", {})

        # Parse the intent to understand what to generate
        generation_type = self._parse_intent(intent)

        # Generate appropriate code
        if generation_type == "api_endpoint":
            code = self._generate_api_endpoint(context)
        elif generation_type == "crud":
            code = self._generate_crud_api(context)
        elif generation_type == "data_model":
            code = self._generate_data_model(context)
        else:
            code = self._generate_generic_api(context)

        return {
            "success": True,
            "generated_code": code,
            "generation_type": generation_type,
            "files_created": self._get_file_list(code),
            "instructions": self._get_implementation_instructions(generation_type),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _parse_intent(self, intent: str) -> str:
        """Parse intent to determine generation type"""
        intent_lower = intent.lower()

        if "crud" in intent_lower:
            return "crud"
        elif "model" in intent_lower or "schema" in intent_lower:
            return "data_model"
        elif "endpoint" in intent_lower or "api" in intent_lower:
            return "api_endpoint"
        else:
            return "generic"

    def _generate_api_endpoint(self, context: Dict) -> str:
        """Generate a single API endpoint"""
        endpoint_name = context.get("endpoint_name", "process")
        function_name = context.get(
            "function_name",
            endpoint_name.replace(
                "-",
                "_"))
        description = context.get(
            "description",
            f"Process {endpoint_name} request")

        # Generate parameter list
        params = context.get("parameters", {})
        if params:
            param_list = []
            for name, ptype in params.items():
                param_list.append(f"{name}: {ptype}")
            parameters = ", ".join(param_list)
        else:
            parameters = "request: Dict"

        # Generate validation logic
        validation_logic = "# Validate required fields\n        "
        if params:
            for param in params:
                validation_logic += f"if not {param}:\n            raise ValidationError('{param} is required')\n        "
        else:
            validation_logic += "pass"

        # Generate processing logic
        processing_logic = """# Main processing logic
        result = {
            'processed': True,
            'data': request
        }"""

        return self.templates["fastapi_endpoint"].format(
            endpoint_name=endpoint_name,
            function_name=function_name,
            parameters=parameters,
            description=description,
            validation_logic=validation_logic,
            processing_logic=processing_logic,
        )

    def _generate_crud_api(self, context: Dict) -> str:
        """Generate CRUD API endpoints"""
        entity = context.get("entity", "item")
        entity_plural = context.get("entity_plural", f"{entity}s")
        model_name = context.get("model_name", f"{entity.capitalize()}Model")

        # Generate model
        model_code = f"""
class {model_name}(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
"""

        # Generate CRUD endpoints
        crud_code = self.templates["crud_operations"].format(
            entity=entity, entity_plural=entity_plural, model_name=model_name
        )

        # Combine into full API
        return self.templates["rest_api"].format(
            title=f"{entity.capitalize()} Management API",
            models=model_code,
            endpoints=crud_code,
        )

    def _generate_data_model(self, context: Dict) -> str:
        """Generate Pydantic data models"""
        model_name = context.get("model_name", "DataModel")
        fields = context.get(
            "fields", {"id": "int", "name": "str", "created_at": "datetime"}
        )

        model_code = f"class {model_name}(BaseModel):\n"
        for field_name, field_type in fields.items():
            optional = "Optional[" if field_name != "id" else ""
            close_bracket = "]" if optional else ""
            default = " = None" if optional else ""
            model_code += (
                f"    {field_name}: {optional}{field_type}{close_bracket}{default}\n")

        return model_code

    def _generate_generic_api(self, context: Dict) -> str:
        """Generate a generic API structure"""
        return self.templates["rest_api"].format(
            title="Generated API",
            models=self._generate_data_model(context),
            endpoints=self._generate_api_endpoint(context),
        )

    def _get_file_list(self, code: str) -> list:
        """Determine which files would be created"""
        files = []
        if "from fastapi import" in code:
            files.append("api_endpoints.py")
        if "class" in code and "BaseModel" in code:
            files.append("models.py")
        return files

    def _get_implementation_instructions(self, generation_type: str) -> str:
        """Get instructions for implementing generated code"""
        instructions = {
            "api_endpoint": "Add this endpoint to your main FastAPI application (mcp/main.py)",
            "crud": "Create a new file 'crud_api.py' and import it in your main application",
            "data_model": "Add this model to your models.py file or create one if it doesn't exist",
            "generic": "Integrate this code into your existing API structure",
        }
        return instructions.get(
            generation_type, "Review and integrate the generated code"
        )


# Export the agent
code_generator_agent = CodeGeneratorAgent()
