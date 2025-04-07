import json
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .agent import agent_executor, execute_tool

class AgentStep:
    def __init__(self, tool_name, tool_args, tool_output):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_output = tool_output
    
    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_output": self.tool_output
        }

@method_decorator(csrf_exempt, name='dispatch')
class InvokeAgentView(View):
    def post(self, request):
        try:
            body = json.loads(request.body)
            content = body.get("content")
            if not content:
                return JsonResponse({"error": "No content provided."}, status=400)
            
            # Agent execution with tracking
            agent_steps = []
            
            # Import the execute_tool function directly from the module
            # and create a wrapper to track steps
            original_execute_tool = execute_tool
            
            def tracking_execute_tool(tool_call):
                tool_name = tool_call.tool_calls[0]["name"]
                tool_args = tool_call.tool_calls[0]["args"]
                tool_message = original_execute_tool(tool_call)
                agent_steps.append(AgentStep(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_output=tool_message.content
                ))
                return tool_message
            
            # Update the module's execute_tool function
            import sys
            module = sys.modules[agent_executor.__module__]
            original_module_execute_tool = module.execute_tool
            module.execute_tool = tracking_execute_tool
            
            try:
                result = agent_executor.invoke(
                    input=content,
                    verbose=True
                )
                return JsonResponse({
                    "answer": result.get("args", {}).get("answer", "No answer found"),
                    "tools_used": result.get("args", {}).get("tools_used", []),
                    "steps": [step.to_dict() for step in agent_steps]
                })
            finally:
                # Restore the original execute_tool function
                module.execute_tool = original_module_execute_tool
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)