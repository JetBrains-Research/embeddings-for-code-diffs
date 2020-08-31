public Node getChildBefore(Node child) {
    if (child == first) {
      return null;
    }
    Node n = first;
    if (n == null) {
      throw new RuntimeException("node is not a child");
    }

    while (n.next != child) {
      n = n.next;
      if (n == null) {
        throw new RuntimeException("node is not a child");
      }
    }
    return n;
  }