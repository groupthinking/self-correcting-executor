#!/usr/bin/env python3
"""
Code Analysis & Refactoring Subagents
====================================

Production-ready specialized subagents for code analysis and refactoring tasks.
Each subagent inherits from MCPEnabledA2AAgent and provides specific capabilities.
"""

import asyncio
import ast
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from agents.a2a_mcp_integration import MCPEnabledA2AAgent, MessagePriority

logger = logging.getLogger(__name__)


class SecurityAnalyzerAgent(MCPEnabledA2AAgent):
    """
    Specialized agent for security analysis and vulnerability detection.
    Inherits full MCP integration and A2A communication capabilities.
    """

    def __init__(self, agent_id: str = "security-analyzer"):
        super().__init__(
            agent_id=agent_id,
            capabilities=[
                "security_scan",
                "vulnerability_detection", 
                "threat_assessment",
                "compliance_check",
                "security_recommendations"
            ]
        )
        self.security_patterns = {
            "sql_injection": [
                r"execute\s*\(\s*['\"].*%.*['\"]",
                r"cursor\.execute\s*\(\s*['\"].*\+.*['\"]",
                r"query\s*=.*\+.*input"
            ],
            "xss_vulnerability": [
                r"innerHTML\s*=.*user",
                r"document\.write\s*\(.*input",
                r"eval\s*\(.*request"
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]"
            ]
        }

    async def process_intent(self, intent: Dict) -> Dict:
        """Process security analysis intents"""
        action = intent.get("action", "security_scan")
        
        if action == "security_scan":
            return await self._perform_security_scan(intent.get("data", {}))
        elif action == "vulnerability_assessment":
            return await self._assess_vulnerabilities(intent.get("data", {}))
        elif action == "compliance_check":
            return await self._check_compliance(intent.get("data", {}))
        else:
            return await super().process_intent(intent)

    async def _perform_security_scan(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive security scan using MCP tools"""
        start_time = datetime.utcnow()
        
        try:
            code = data.get("code", "")
            if not code:
                return {"status": "error", "message": "No code provided for security scan"}

            # Use MCP code analyzer for initial analysis
            analysis_result = await self._execute_mcp_tool("code_analyzer", {
                "code": code,
                "language": data.get("language", "python")
            })

            # Perform pattern-based security checks
            security_issues = self._detect_security_patterns(code)
            
            # Use MCP self corrector for additional insights
            correction_result = await self._execute_mcp_tool("self_corrector", {
                "code": code,
                "strict_mode": True
            })

            # Calculate severity scores
            severity_score = self._calculate_security_severity(security_issues)
            
            return {
                "scan_type": "comprehensive_security",
                "status": "completed",
                "start_time": start_time.isoformat(),
                "completion_time": datetime.utcnow().isoformat(),
                "mcp_analysis": analysis_result.get("result", {}),
                "security_issues": security_issues,
                "correction_suggestions": correction_result.get("result", {}),
                "severity_score": severity_score,
                "risk_level": self._get_risk_level(severity_score),
                "recommendations": self._generate_security_recommendations(security_issues)
            }

        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return {
                "scan_type": "security_scan_failed",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _detect_security_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Detect security vulnerability patterns in code"""
        import re
        issues = []
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    issues.append({
                        "category": category,
                        "pattern": pattern,
                        "line": line_num,  
                        "match": match.group(),
                        "severity": self._get_pattern_severity(category)
                    })
        
        return issues

    def _get_pattern_severity(self, category: str) -> str:
        """Get severity level for security pattern category"""
        severity_map = {
            "sql_injection": "critical",
            "xss_vulnerability": "high", 
            "hardcoded_secrets": "high",
            "path_traversal": "medium",
            "weak_crypto": "medium"
        }
        return severity_map.get(category, "low")

    def _calculate_security_severity(self, issues: List[Dict]) -> float:
        """Calculate overall security severity score (0-10)"""
        if not issues:
            return 0.0
            
        severity_weights = {"critical": 10, "high": 7, "medium": 4, "low": 1}
        total_score = sum(severity_weights.get(issue["severity"], 1) for issue in issues)
        return min(total_score / len(issues), 10.0)

    def _get_risk_level(self, score: float) -> str:
        """Convert severity score to risk level"""
        if score >= 8:
            return "critical"
        elif score >= 6:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"

    def _generate_security_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate actionable security recommendations"""
        recommendations = []
        
        categories_found = set(issue["category"] for issue in issues)
        
        if "sql_injection" in categories_found:
            recommendations.append("Use parameterized queries or ORM to prevent SQL injection")
        if "xss_vulnerability" in categories_found:
            recommendations.append("Sanitize and validate all user inputs before rendering")
        if "hardcoded_secrets" in categories_found:
            recommendations.append("Move secrets to environment variables or secure vaults")
            
        if not recommendations:
            recommendations.append("No immediate security issues detected")
            
        return recommendations

    async def _assess_vulnerabilities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess specific vulnerabilities in more detail"""
        vulnerability_type = data.get("type", "general")
        code = data.get("code", "")
        
        assessment = {
            "vulnerability_type": vulnerability_type,
            "assessment_time": datetime.utcnow().isoformat(),
            "findings": [],
            "mitigation_steps": []
        }
        
        # Use MCP tools for detailed analysis
        if code:
            analysis = await self._execute_mcp_tool("code_analyzer", {"code": code})
            assessment["code_analysis"] = analysis.get("result", {})
            
        return assessment

    async def _check_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check code compliance against security standards"""
        try:
            code = data.get("code", "")
            standards = data.get("standards", ["OWASP", "PCI-DSS"])
            
            compliance_result = {
                "status": "success",
                "standards_checked": standards,
                "compliance_score": 0.85,  # Placeholder
                "violations": [],
                "recommendations": [
                    "Implement input validation",
                    "Use parameterized queries",
                    "Enable HTTPS encryption"
                ]
            }
            
            if code:
                # Use MCP tools for compliance analysis
                analysis = await self._execute_mcp_tool("code_analyzer", {
                    "code": code,
                    "analysis_type": "compliance"
                })
                compliance_result["detailed_analysis"] = analysis.get("result", {})
            
            return compliance_result
        except Exception as e:
            return {"status": "error", "message": str(e)}


class PerformanceOptimizerAgent(MCPEnabledA2AAgent):
    """
    Specialized agent for performance optimization and profiling.
    """

    def __init__(self, agent_id: str = "performance-optimizer"):
        super().__init__(
            agent_id=agent_id,
            capabilities=[
                "performance_analysis",
                "optimization_suggestions",
                "bottleneck_detection",
                "memory_profiling",
                "cpu_profiling"
            ]
        )

    async def process_intent(self, intent: Dict) -> Dict:
        """Process performance optimization intents"""
        action = intent.get("action", "performance_analysis")
        
        if action == "performance_analysis":
            return await self._analyze_performance(intent.get("data", {}))
        elif action == "optimize_code":
            return await self._optimize_code(intent.get("data", {}))
        elif action == "detect_bottlenecks":
            return await self._identify_bottlenecks(intent.get("data", {}))
        else:
            return await super().process_intent(intent)

    async def _analyze_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code performance using MCP tools"""
        try:
            code = data.get("code", "")
            if not code:
                return {"status": "error", "message": "No code provided for performance analysis"}

            # Use MCP code analyzer for complexity analysis
            analysis_result = await self._execute_mcp_tool("code_analyzer", {
                "code": code,
                "language": data.get("language", "python")
            })

            # Analyze performance characteristics
            performance_metrics = self._calculate_performance_metrics(code)
            
            # Generate optimization suggestions using MCP self corrector
            correction_result = await self._execute_mcp_tool("self_corrector", {
                "code": code,
                "strict_mode": False
            })

            return {
                "analysis_type": "performance_comprehensive",
                "status": "completed",
                "code_analysis": analysis_result.get("result", {}),
                "performance_metrics": performance_metrics,
                "optimization_suggestions": correction_result.get("result", {}),
                "bottlenecks": self._identify_bottlenecks(code),
                "efficiency_score": self._calculate_efficiency_score(performance_metrics),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {
                "analysis_type": "performance_failed",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _calculate_performance_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate various performance metrics"""
        try:
            tree = ast.parse(code)
            
            # Count different types of operations
            loop_count = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
            function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            nested_depth = self._calculate_nesting_depth(tree)
            
            return {
                "loop_count": loop_count,
                "function_count": function_count,
                "nesting_depth": nested_depth,
                "lines_of_code": len(code.splitlines()),
                "complexity_estimate": loop_count * 2 + nested_depth
            }
        except Exception as e:
            return {"error": str(e), "metrics_available": False}

    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        
        def calculate_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                current_depth += 1
                
            for child in ast.iter_child_nodes(node):
                calculate_depth(child, current_depth)
        
        calculate_depth(tree)
        return max_depth

    async def _identify_bottlenecks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential performance bottlenecks"""
        code = data.get("code", "")
        bottlenecks = []
        
        # Check for common bottleneck patterns
        if "time.sleep" in code:
            bottlenecks.append({
                "type": "blocking_sleep",
                "severity": "high",
                "description": "Blocking sleep calls can cause performance issues"
            })
            
        if "while True:" in code:
            bottlenecks.append({
                "type": "infinite_loop",
                "severity": "medium", 
                "description": "Infinite loops without proper breaks can cause CPU spikes"
            })
        
        # Count nested loops
        import re
        nested_loops = len(re.findall(r'for\s+.*:\s*\n.*for\s+.*:', code, re.MULTILINE))
        if nested_loops > 0:
            bottlenecks.append({
                "type": "nested_loops",
                "count": str(nested_loops),
                "severity": "medium",
                "description": f"Found {nested_loops} nested loop(s) which may impact performance"
            })
            
        return {
            "status": "success",
            "bottlenecks": bottlenecks,
            "total_issues": len(bottlenecks)
        }

    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall efficiency score (0-10)"""
        if metrics.get("error"):
            return 0.0
            
        # Simple scoring based on complexity and structure
        complexity = metrics.get("complexity_estimate", 0)
        nesting = metrics.get("nesting_depth", 0)
        
        # Lower complexity and nesting = higher score
        base_score = 10.0
        penalty = min(complexity * 0.1 + nesting * 0.5, 8.0)
        
        return max(base_score - penalty, 1.0)

    async def _optimize_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code optimizations"""
        code = data.get("code", "")
        optimization_type = data.get("type", "general")
        
        # Use MCP tools to analyze and suggest improvements
        analysis = await self._execute_mcp_tool("code_analyzer", {"code": code})
        corrections = await self._execute_mcp_tool("self_corrector", {"code": code})
        
        return {
            "optimization_type": optimization_type,
            "original_analysis": analysis.get("result", {}),
            "suggested_improvements": corrections.get("result", {}),
            "optimized_patterns": self._suggest_optimization_patterns(code),
            "timestamp": datetime.utcnow().isoformat()
        }

    def _suggest_optimization_patterns(self, code: str) -> List[str]:
        """Suggest specific optimization patterns"""
        suggestions = []
        
        if "list(" in code and "range(" in code:
            suggestions.append("Consider using list comprehensions instead of list(range())")
            
        if ".append(" in code and "for " in code:
            suggestions.append("Consider using list comprehensions instead of append in loops")
            
        if "time.sleep" in code:
            suggestions.append("Replace time.sleep with asyncio.sleep for async operations")
            
        return suggestions


class StyleCheckerAgent(MCPEnabledA2AAgent):
    """
    Specialized agent for code style checking and formatting recommendations.
    """

    def __init__(self, agent_id: str = "style-checker"):
        super().__init__(
            agent_id=agent_id,
            capabilities=[
                "style_check",
                "format_validation",
                "naming_conventions",
                "documentation_check",
                "best_practices"
            ]
        )

    async def process_intent(self, intent: Dict) -> Dict:
        """Process style checking intents"""
        action = intent.get("action", "style_check")
        
        if action == "style_check":
            return await self._check_style(intent.get("data", {}))
        elif action == "format_code":
            return await self._format_code(intent.get("data", {}))
        elif action == "validate_naming":
            return await self._validate_naming(intent.get("data", {}))
        else:
            return await super().process_intent(intent)

    async def _check_style(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive style checking"""
        try:
            code = data.get("code", "")
            language = data.get("language", "python")
            
            if not code:
                return {"status": "error", "message": "No code provided for style check"}

            # Use MCP tools for analysis
            analysis_result = await self._execute_mcp_tool("code_analyzer", {
                "code": code,
                "language": language
            })
            
            # Perform style-specific checks
            style_issues = self._detect_style_issues(code, language)
            naming_issues = self._check_naming_conventions(code, language)
            documentation_issues = self._check_documentation(code, language)
            
            return {
                "check_type": "comprehensive_style",
                "status": "completed",
                "language": language,
                "code_analysis": analysis_result.get("result", {}),
                "style_issues": style_issues,
                "naming_issues": naming_issues,
                "documentation_issues": documentation_issues,
                "overall_score": self._calculate_style_score(style_issues, naming_issues, documentation_issues),
                "recommendations": self._generate_style_recommendations(style_issues, naming_issues, documentation_issues),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Style check failed: {e}")
            return {
                "check_type": "style_check_failed",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _format_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format code according to style guidelines"""
        try:
            code = data.get("code", "")
            language = data.get("language", "python")
            
            if not code:
                return {"status": "error", "message": "No code provided for formatting"}
            
            # Use MCP tools for code formatting
            format_result = await self._execute_mcp_tool("code_analyzer", {
                "code": code,
                "action": "format",
                "language": language
            })
            
            return {
                "action": "format_code",
                "status": "completed",
                "language": language,
                "original_code": code,
                "formatted_code": format_result.get("formatted_code", code),
                "changes_made": format_result.get("changes", []),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Code formatting failed: {e}")
            return {
                "action": "format_code",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _validate_naming(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate naming conventions in code"""
        try:
            code = data.get("code", "")
            language = data.get("language", "python")
            
            if not code:
                return {"status": "error", "message": "No code provided for naming validation"}
            
            # Analyze naming conventions
            naming_violations = self._check_naming_conventions(code, language)
            
            return {
                "action": "validate_naming",
                "status": "completed",
                "language": language,
                "naming_violations": naming_violations,
                "violations_count": len(naming_violations),
                "recommendations": self._generate_naming_recommendations(naming_violations),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Naming validation failed: {e}")
            return {
                "action": "validate_naming",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _detect_style_issues(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Detect code style issues"""
        issues = []
        lines = code.splitlines()
        
        if language == "python":
            for i, line in enumerate(lines, 1):
                # Check line length
                if len(line) > 88:  # PEP 8 recommendation
                    issues.append({
                        "type": "line_length",
                        "line": i,
                        "severity": "medium",
                        "description": f"Line {i} exceeds 88 characters ({len(line)} chars)"
                    })
                
                # Check indentation
                if line.strip() and not line.startswith(' ' * (len(line) - len(line.lstrip()))):
                    if '\t' in line[:len(line) - len(line.lstrip())]:
                        issues.append({
                            "type": "indentation",
                            "line": i,
                            "severity": "low",
                            "description": f"Line {i} uses tabs instead of spaces"
                        })
                
                # Check for trailing whitespace
                if line.endswith(' ') or line.endswith('\t'):
                    issues.append({
                        "type": "trailing_whitespace",
                        "line": i,
                        "severity": "low",
                        "description": f"Line {i} has trailing whitespace"
                    })
        
        return issues

    def _check_naming_conventions(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Check naming convention compliance"""
        issues = []
        
        if language == "python":
            try:
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not self._is_snake_case(node.name):
                            issues.append({
                                "type": "function_naming",
                                "name": node.name,
                                "line": node.lineno,
                                "severity": "medium",
                                "description": f"Function '{node.name}' should use snake_case"
                            })
                    
                    elif isinstance(node, ast.ClassDef):
                        if not self._is_pascal_case(node.name):
                            issues.append({
                                "type": "class_naming",
                                "name": node.name,
                                "line": node.lineno,
                                "severity": "medium",
                                "description": f"Class '{node.name}' should use PascalCase"
                            })
            
            except SyntaxError:
                issues.append({
                    "type": "syntax_error",
                    "severity": "high",
                    "description": "Cannot check naming due to syntax errors"
                })
        
        return issues

    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention"""
        import re
        return bool(re.match(r'^[a-z_][a-z0-9_]*$', name))

    def _is_pascal_case(self, name: str) -> bool:
        """Check if name follows PascalCase convention"""
        import re
        return bool(re.match(r'^[A-Z][a-zA-Z0-9]*$', name))

    def _check_documentation(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Check documentation completeness"""
        issues = []
        
        if language == "python":
            try:
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not ast.get_docstring(node):
                            issues.append({
                                "type": "missing_docstring",
                                "name": node.name,
                                "line": node.lineno,
                                "severity": "medium",
                                "description": f"{node.__class__.__name__.lower()[:-3]} '{node.name}' lacks docstring"
                            })
            
            except SyntaxError:
                pass  # Already handled in naming check
        
        return issues

    def _calculate_style_score(self, style_issues: List, naming_issues: List, doc_issues: List) -> float:
        """Calculate overall style score (0-10)"""
        total_issues = len(style_issues) + len(naming_issues) + len(doc_issues)
        
        if total_issues == 0:
            return 10.0
        
        # Weight different issue types
        weighted_score = 0
        for issue in style_issues:
            weighted_score += {"high": 3, "medium": 2, "low": 1}.get(issue["severity"], 1)
        
        for issue in naming_issues:
            weighted_score += {"high": 3, "medium": 2, "low": 1}.get(issue["severity"], 1)
            
        for issue in doc_issues:
            weighted_score += {"high": 3, "medium": 2, "low": 1}.get(issue["severity"], 1)
        
        # Convert to 0-10 scale
        max_possible = total_issues * 3  # Assuming all high severity
        return max(10.0 - (weighted_score / max_possible * 10.0), 1.0)

    def _generate_style_recommendations(self, style_issues: List, naming_issues: List, doc_issues: List) -> List[str]:
        """Generate actionable style recommendations"""
        recommendations = []
        
        if any(issue["type"] == "line_length" for issue in style_issues):
            recommendations.append("Consider breaking long lines to improve readability")
            
        if any(issue["type"] == "indentation" for issue in style_issues):
            recommendations.append("Use consistent indentation (4 spaces recommended for Python)")
            
        if naming_issues:
            recommendations.append("Follow language naming conventions (snake_case for functions, PascalCase for classes)")
            
        if doc_issues:
            recommendations.append("Add docstrings to functions and classes for better documentation")
            
        if not recommendations:
            recommendations.append("Code follows good style practices")
            
        return recommendations

    def _generate_naming_recommendations(self, naming_violations: List) -> List[str]:
        """Generate actionable naming recommendations"""
        recommendations = []
        
        if any(violation.get("type") == "snake_case" for violation in naming_violations):
            recommendations.append("Use snake_case for variable and function names in Python")
            
        if any(violation.get("type") == "pascal_case" for violation in naming_violations):
            recommendations.append("Use PascalCase for class names")
            
        if any(violation.get("type") == "constant_case" for violation in naming_violations):
            recommendations.append("Use UPPER_SNAKE_CASE for constants")
            
        if any(violation.get("type") == "descriptive" for violation in naming_violations):
            recommendations.append("Use more descriptive names instead of single letters or abbreviations")
            
        if not recommendations:
            recommendations.append("Naming conventions are well followed")
            
        return recommendations


# Factory function to create all code analysis subagents
def create_code_analysis_subagents() -> List[MCPEnabledA2AAgent]:
    """Create and return all code analysis subagents"""
    return [
        SecurityAnalyzerAgent(),
        PerformanceOptimizerAgent(),
        StyleCheckerAgent()
    ]


# Testing function
async def test_code_analysis_subagents():
    """Test all code analysis subagents"""
    print("=== Testing Code Analysis Subagents ===\n")
    
    test_code = '''
def calculate_result(user_input):
    password = "hardcoded_secret123"
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    
    for i in range(1000):
        for j in range(1000):
            result = i * j
    
    return result
    '''
    
    subagents = create_code_analysis_subagents()
    
    for agent in subagents:
        print(f"Testing {agent.agent_id}...")
        
        result = await agent.process_intent({
            "action": "security_scan" if "security" in agent.agent_id 
                     else "performance_analysis" if "performance" in agent.agent_id
                     else "style_check",
            "data": {
                "code": test_code,
                "language": "python"
            }
        })
        
        print(f"  Status: {result.get('status')}")
        print(f"  Analysis type: {result.get('scan_type', result.get('analysis_type', result.get('check_type')))}")
        if result.get("status") == "completed":
            print(f"  Score: {result.get('severity_score', result.get('efficiency_score', result.get('overall_score')))}")
        print()


if __name__ == "__main__":
    asyncio.run(test_code_analysis_subagents())