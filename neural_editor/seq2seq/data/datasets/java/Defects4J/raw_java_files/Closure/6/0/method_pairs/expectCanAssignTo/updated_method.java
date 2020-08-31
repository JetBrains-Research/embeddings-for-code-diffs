
  boolean expectCanAssignTo(NodeTraversal t, Node n, JSType rightType,
      JSType leftType, String msg) {
    if (!rightType.canAssignTo(leftType)) {
      mismatch(t, n, msg, rightType, leftType);
      return false;
    }
    return true;
  }