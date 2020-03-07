"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""
import ast

# Constants used as both edge and context labels.
BREAK = "break"
CONTINUE = "continue"
NEXT = "next"
RAISE = "raise"
RETURN = "return"
RETURN_VALUE = "return_value"

# Constants used only as edge labels.
IF = "if"
ELSE = "else"
ENTER = "enter"
MATCH = "match"
NO_MATCH = "no_match"


class CFNode:
    """
    A node on the control flow graph.
    """


class CFGraph:
    """
    The control-flow graph.

    This is a directed graph (not necessarily acyclic) with labelled edges.
    Most nodes will correspond directly to an AST statement.
    """

    def __init__(self):
        # Representation: nodes form a set; for each node, edges[node]
        # is a mapping from labels to nodes.
        self._nodes = set()
        self._edges = {}

        # Back-edges: mapping from each node to the *set* of edges that
        # enter it. Edges are characterised as pairs (node, label).
        self._backedges = {}

        # We'll usually want some named nodes. (For example, for a graph
        # representing the control flow in a function, we'll want to mark the
        # entry and exit nodes.) Those go in the dictionary below.
        self.context = {}

    # Graph interface

    def add_node(self, node):
        """
        Add a node to the graph. Raises on an attempt to add a node that
        already exists.
        """
        assert node not in self._nodes
        self._nodes.add(node)
        self._edges[node] = {}
        self._backedges[node] = set()

    def remove_node(self, node):
        """
        Remove a node from the graph. Fails if there are edges to or
        from that node.
        """
        assert not self._backedges[node]
        assert not self._edges[node]
        self._nodes.remove(node)

    def add_edge(self, source, label, target):
        """
        Add a labelled edge to the graph. Raises if an edge from the given
        source, with the given label, already exists.
        """
        assert label not in self._edges[source]
        self._edges[source][label] = target

        assert (source, label) not in self._backedges[target]
        self._backedges[target].add((source, label))

    def remove_edge(self, source, label, target):
        self._backedges[target].remove((source, label))
        self._edges[source].pop(label)

    def edge(self, source, label):
        """
        Get the target of a given edge.
        """
        return self._edges[source][label]

    def edge_labels(self, source):
        """
        Get labels of all edges.
        """
        return set(self._edges[source].keys())

    def edges_to(self, target):
        """
        Set of pairs (source, label) representing edges to this node.
        """
        return self._backedges[target]

    # Analysis interface

    def cfnode(self, edges):
        """
        Create a new control-flow node and add it to the graph.

        Returns the newly-created node.
        """
        node = CFNode()
        self.add_node(node)
        for name, target in edges.items():
            self.add_edge(node, name, target)
        return node

    def _analyse_declaration(
        self, statement: ast.Global, context: dict
    ) -> CFNode:
        """
        Analyse a declaration (a global or nonlocal statement).

        These statements are non-executable, so we don't create a node
        for them. Instead, we simply return the NEXT node from the context.
        """
        return context[NEXT]

    def _analyse_generic(self, statement: ast.stmt, context: dict) -> CFNode:
        """
        Analyse a generic statement that doesn't affect control flow.
        """
        return self.cfnode({RAISE: context[RAISE], NEXT: context[NEXT]})

    def _analyse_loop(self, statement: ast.stmt, context: dict) -> CFNode:
        """
        Analyse a loop statement (for or while).
        """
        loop_node = self.cfnode(
            {
                RAISE: context[RAISE],
                ELSE: self.analyse_statements(statement.orelse, context),
                # The target for the ENTER edge is created below.
            }
        )

        body_context = context.copy()
        body_context[BREAK] = context[NEXT]
        body_context[CONTINUE] = loop_node
        body_context[NEXT] = loop_node
        body_node = self.analyse_statements(statement.body, body_context)

        self.add_edge(loop_node, ENTER, body_node)
        return loop_node

    def _analyse_try_except_else(
        self, statement: ast.Try, context: dict
    ) -> CFNode:
        """
        Analyse the try-except-else part of a try statement. The finally
        part is ignored, as though it weren't present.
        """
        # Process handlers backwards; raise if the last handler doesn't match.
        raise_node = context[RAISE]
        for handler in reversed(statement.handlers):
            match_node = self.analyse_statements(handler.body, context)
            if handler.type is None:
                # Bare except always matches, never raises.
                raise_node = match_node
            else:
                raise_node = self.cfnode(
                    {
                        RAISE: context[RAISE],
                        MATCH: match_node,
                        NO_MATCH: raise_node,
                    }
                )

        body_context = context.copy()
        body_context[RAISE] = raise_node
        body_context[NEXT] = self.analyse_statements(statement.orelse, context)
        body_node = self.analyse_statements(statement.body, body_context)

        return self.cfnode({ENTER: body_node})

    def analyse_Assert(self, statement: ast.Assert, context: dict) -> CFNode:
        """
        Analyse an assert statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_Assign(self, statement: ast.Assign, context: dict) -> CFNode:
        """
        Analyse an assignment statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_AugAssign(
        self, statement: ast.AugAssign, context: dict
    ) -> CFNode:
        """
        Analyse an augmented assignment statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_Break(self, statement: ast.Break, context: dict) -> CFNode:
        """
        Analyse a break statement.
        """
        return self.cfnode({BREAK: context[BREAK]})

    def analyse_ClassDef(
        self, statement: ast.ClassDef, context: dict
    ) -> CFNode:
        """
        Analyse a class definition.
        """
        return self._analyse_generic(statement, context)

    def analyse_Continue(
        self, statement: ast.Continue, context: dict
    ) -> CFNode:
        """
        Analyse a continue statement.
        """
        return self.cfnode({CONTINUE: context[CONTINUE]})

    def analyse_Delete(self, statement: ast.Delete, context: dict) -> CFNode:
        """
        Analyse a del statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_Expr(self, statement: ast.Expr, context: dict) -> CFNode:
        """
        Analyse an expression (used as a statement).
        """
        return self._analyse_generic(statement, context)

    def analyse_For(self, statement: ast.For, context: dict) -> CFNode:
        """
        Analyse a for statement.
        """
        return self._analyse_loop(statement, context)

    def analyse_FunctionDef(
        self, statement: ast.FunctionDef, context: dict
    ) -> CFNode:
        """
        Analyse a function definition.
        """
        return self._analyse_generic(statement, context)

    def analyse_Global(self, statement: ast.Global, context: dict) -> CFNode:
        """
        Analyse a global statement
        """
        return self._analyse_declaration(statement, context)

    def analyse_If(self, statement: ast.If, context: dict) -> CFNode:
        """
        Analyse an if statement.
        """
        return self.cfnode(
            {
                IF: self.analyse_statements(statement.body, context),
                ELSE: self.analyse_statements(statement.orelse, context),
                RAISE: context[RAISE],
            }
        )

    def analyse_Import(self, statement: ast.Import, context: dict) -> CFNode:
        """
        Analyse an import statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_ImportFrom(
        self, statement: ast.ImportFrom, context: dict
    ) -> CFNode:
        """
        Analyse a from ... import statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_Nonlocal(
        self, statement: ast.Nonlocal, context: dict
    ) -> CFNode:
        """
        Analyse a nonlocal statement
        """
        return self._analyse_declaration(statement, context)

    def analyse_Pass(self, statement: ast.Pass, context: dict) -> CFNode:
        """
        Analyse a pass statement.
        """
        return self.cfnode({NEXT: context[NEXT]})

    def analyse_Raise(self, statement: ast.Raise, context: dict) -> CFNode:
        """
        Analyse a raise statement.
        """
        return self.cfnode({RAISE: context[RAISE]})

    def analyse_Return(self, statement: ast.Return, context: dict) -> CFNode:
        """
        Analyse a return statement.
        """
        if statement.value is None:
            return self.cfnode({RETURN: context[RETURN]})
        else:
            return self.cfnode(
                {RAISE: context[RAISE], RETURN_VALUE: context[RETURN_VALUE]},
            )

    def analyse_Try(self, statement: ast.Try, context: dict) -> CFNode:
        """
        Analyse a try statement.
        """
        # To analyse a try-except-else-finally block, it's easier to think
        # of it as two separate pieces: it's equivalent to the try-except-else
        # piece, nested *inside* a try-finally.

        # The finally block can be entered by various different means (from
        # a return, raise, break or continue in the try-except-else part, or
        # from completing the try-except-else without error). For each of
        # these different entrances to the finally block, the target on
        # *leaving* the finally block may be different. So we will
        # construct different paths in the control flow graph for each of
        # the possible leave targets for the finally block. (This is similar
        # to the way that Python >= 3.9 bytecode handles the finally.)

        # We also want to avoid generating *all* six possible finally paths:
        # some of them may not be relevant. So we adopt the following approach:

        # 1. Create dummy target nodes for the try-except-else, and analyse
        #    the try-except-else with those dummy nodes.
        # 2. For only the dummy nodes that are actually reached, construct
        #    the corresponding finally paths.
        # 3. Link up the graphs.

        # For each actual node in the context (excluding duplicates),
        # create a corresponding dummy node.
        dummy_nodes = {}
        for node in set(context.values()):
            dummy_nodes[node] = self.cfnode({})

        # Analyse the try-except-else part of the statement using those dummy
        # nodes.
        try_except_else_context = {
            key: dummy_nodes[node] for key, node in context.items()
        }

        entry_node = self._analyse_try_except_else(
            statement, try_except_else_context
        )

        # Now iterate through the dummy nodes. For those that aren't reached,
        # remove them. For those that are, replace with the corresponding
        # finally code. Note that there will always be at least one of the
        # dummy nodes reachable from the try-except-else, so we'll always
        # analyse the finally code at least once.

        for end_node, dummy_node in dummy_nodes.items():
            if self.edges_to(dummy_node):
                # Dummy node is reachable from the try-except-else.
                finally_context = context.copy()
                finally_context[NEXT] = end_node
                finally_node = self.analyse_statements(
                    statement.finalbody, finally_context
                )

                # Make all edges to the dummy node point to the target node.
                for source, label in list(self.edges_to(dummy_node)):
                    self.remove_edge(source, label, dummy_node)
                    self.add_edge(source, label, finally_node)

            self.remove_node(dummy_node)

        return entry_node

    def analyse_While(self, statement: ast.While, context: dict) -> CFNode:
        """
        Analyse a while statement
        """
        return self._analyse_loop(statement, context)

    def analyse_With(self, statement: ast.With, context: dict) -> CFNode:
        """
        Analyse a with statement.
        """
        return self.cfnode(
            {
                ENTER: self.analyse_statements(statement.body, context),
                RAISE: context[RAISE],
            }
        )

    def analyse_statements(self, statements: list, context: dict) -> CFNode:
        """
        Analyse a sequence of statements.
        """
        next = context[NEXT]
        for statement in reversed(statements):
            statement_context = context.copy()
            statement_context[NEXT] = next

            method_name = "analyse_" + type(statement).__name__
            analyser = getattr(self, method_name)
            next = analyser(statement, statement_context)

        return next

    @classmethod
    def from_function(cls, ast_node):
        """
        Construct a control flow graph for an AST FunctionDef node.

        Parameters
        ----------
        ast_node : ast.FunctionDef
            Function node in the ast tree of the code being analysed.
        """
        self = cls()
        # Node for returns without an explicit return value.
        return_node = self.cfnode({})
        # Node for returns *with* an explicit return value (which could
        # be None).
        return_value_node = self.cfnode({})
        # Node for exit via raise.
        raise_node = self.cfnode({})

        body_context = {
            NEXT: return_node,
            RAISE: raise_node,
            RETURN: return_node,
            RETURN_VALUE: return_value_node,
        }
        enter_node = self.analyse_statements(ast_node.body, body_context)

        self.context = {
            ENTER: enter_node,
            RAISE: raise_node,
            RETURN: return_node,
            RETURN_VALUE: return_value_node,
        }
        return self
