@Override
  public void process(Node externs, Node root) {
    (new NodeTraversal(compiler, this)).traverse(root);
  }