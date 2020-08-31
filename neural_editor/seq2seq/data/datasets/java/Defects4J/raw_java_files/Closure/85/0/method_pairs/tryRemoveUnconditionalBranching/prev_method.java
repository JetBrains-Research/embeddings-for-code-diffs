
  @SuppressWarnings("fallthrough")
  private Node tryRemoveUnconditionalBranching(Node n) {
    /*
     * For each of the unconditional branching control flow node, check to see
     * if the ControlFlowAnalysis.computeFollowNode of that node is same as
     * the branching target. If it is, the branch node is safe to be removed.
     *
     * This is not as clever as MinimizeExitPoints because it doesn't do any
     * if-else conversion but it handles more complicated switch statements
     * much nicer.
     */

    // If n is null the target is the end of the function, nothing to do.
    if (n == null) {
       return n;
    }

    DiGraphNode<Node, Branch> gNode = curCfg.getDirectedGraphNode(n);

    if (gNode == null) {
      return n;
    }

    if (n.getParent() == null) {
      List<DiGraphEdge<Node,Branch>> outEdges = gNode.getOutEdges();
      if (outEdges.size() == 1) {
        return tryRemoveUnconditionalBranching(outEdges.get(0).getDestination().getValue());
      }
    }
    switch (n.getType()) {
      case Token.BLOCK:
        if (n.hasChildren()) {
          Node first = n.getFirstChild();
          return tryRemoveUnconditionalBranching(first);
        } else {
          return tryRemoveUnconditionalBranching(ControlFlowAnalysis.computeFollowNode(n));
        }
      case Token.RETURN:
        if (n.hasChildren()) {
          break;
        }
      case Token.BREAK:
      case Token.CONTINUE:

        // We are looking for a control flow changing statement that always
        // branches to the same node. If removing it the control flow still
        // branches to that same node. It is safe to remove it.
        List<DiGraphEdge<Node,Branch>> outEdges = gNode.getOutEdges();
        if (outEdges.size() == 1 &&
            // If there is a next node, there is no chance this jump is useless.
            (n.getNext() == null || n.getNext().getType() == Token.FUNCTION)) {

          Preconditions.checkState(outEdges.get(0).getValue() == Branch.UNCOND);
          Node fallThrough = tryRemoveUnconditionalBranching(computeFollowing(n));
          Node nextCfgNode = outEdges.get(0).getDestination().getValue();
          if (nextCfgNode == fallThrough) {
            removeDeadExprStatementSafely(n);
            return fallThrough;
          }
        }
    }
    return n;
  }