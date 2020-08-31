@Override
  public void process(Node externs, Node root) {
    NodeTraversal.traverse(compiler, root, this);
    removeDuplicateDeclarations(root);
    if (MAKE_LOCAL_NAMES_UNIQUE) {
      MakeDeclaredNamesUnique renamer = new MakeDeclaredNamesUnique();
      NodeTraversal t = new NodeTraversal(compiler, renamer);
      t.traverseRoots(externs, root);
    }
    new PropogateConstantAnnotations(compiler, assertOnChange)
        .process(externs, root);
  }