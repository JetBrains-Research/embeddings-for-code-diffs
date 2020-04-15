
  private void computeDependence(final Definition def, Node rValue) {
    NodeTraversal.traverse(compiler, rValue,
        new AbstractCfgNodeTraversalCallback() {
      @Override
      public void visit(NodeTraversal t, Node n, Node parent) {
        if (n.isName() && jsScope.isDeclared(n.getString(), true)) {
          Var dep = jsScope.getVar(n.getString());
            def.depends.add(dep);
        }
      }
    });
  }