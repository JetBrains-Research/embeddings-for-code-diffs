
  private void computeDependence(final Definition def, Node rValue) {
    NodeTraversal.traverse(compiler, rValue,
        new AbstractCfgNodeTraversalCallback() {
      @Override
      public void visit(NodeTraversal t, Node n, Node parent) {
        if (n.isName()) {
          Var dep = jsScope.getVar(n.getString());
          if (dep == null) {
            def.unknownDependencies = true;
          } else {
            def.depends.add(dep);
          }
        }
      }
    });
  }