
  private void interpretAssigns() {
    boolean changes = false;
    do {
      changes = false;

      // We can't use traditional iterators and iterables for this list,
      // because our lazily-evaluated continuations will modify it while
      // we traverse it.
      for (int current = 0; current < maybeUnreferenced.size(); current++) {
        Var var = maybeUnreferenced.get(current);
        if (referenced.contains(var)) {
          maybeUnreferenced.remove(current);
          current--;
        } else {
          boolean assignedToUnknownValue = false;
          boolean hasPropertyAssign = false;

          if (var.getParentNode().isVar() &&
              !NodeUtil.isForIn(var.getParentNode().getParent())) {
            Node value = var.getInitialValue();
            assignedToUnknownValue = value != null &&
                !NodeUtil.isLiteralValue(value, true);
          } else {
            // This was initialized to a function arg or a catch param
            // or a for...in variable.
            assignedToUnknownValue = true;
          }

          for (Assign assign : assignsByVar.get(var)) {
            if (assign.isPropertyAssign) {
              hasPropertyAssign = true;
            } else if (!NodeUtil.isLiteralValue(
                assign.assignNode.getLastChild(), true)) {
              assignedToUnknownValue = true;
            }
          }

          if (assignedToUnknownValue && hasPropertyAssign) {
            changes = markReferencedVar(var) || changes;
            maybeUnreferenced.remove(current);
            current--;
          }
        }
      }
    } while (changes);
  }