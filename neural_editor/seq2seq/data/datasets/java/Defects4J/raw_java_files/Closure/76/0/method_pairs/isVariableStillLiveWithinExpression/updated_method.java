
  private boolean isVariableStillLiveWithinExpression(
      Node n, Node exprRoot, String variable) {
    while (n != exprRoot) {
      VariableLiveness state = VariableLiveness.MAYBE_LIVE;
      switch (n.getParent().getType()) {
        case Token.OR:
        case Token.AND:
          // If the currently node is the first child of
          // AND/OR, be conservative only consider the READs
          // of the second operand.
          if (n.getNext() != null) {
            state = isVariableReadBeforeKill(
                n.getNext(), variable);
            if (state == VariableLiveness.KILL) {
              state = VariableLiveness.MAYBE_LIVE;
            }
          }
          break;

        case Token.HOOK:
          // If current node is the condition, check each following
          // branch, otherwise it is a conditional branch and the
          // other branch can be ignored.
          if (n.getNext() != null && n.getNext().getNext() != null) {
            state = checkHookBranchReadBeforeKill(
                n.getNext(), n.getNext().getNext(), variable);
          }
          break;

        default:
          for(Node sibling = n.getNext(); sibling != null;
              sibling = sibling.getNext()) {
            state = isVariableReadBeforeKill(sibling, variable);
            if (state != VariableLiveness.MAYBE_LIVE) {
              break;
            }
          }
      }

      // If we see a READ or KILL there is no need to continue.
      if (state == VariableLiveness.READ) {
        return true;
      } else if (state == VariableLiveness.KILL) {
        return false;
      }
      n = n.getParent();
    }
    return false;
  }