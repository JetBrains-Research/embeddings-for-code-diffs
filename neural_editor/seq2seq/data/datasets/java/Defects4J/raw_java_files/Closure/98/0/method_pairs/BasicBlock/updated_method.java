
    BasicBlock(BasicBlock parent, Node root) {
      this.parent = parent;

      // only named functions may be hoisted.
      this.isHoisted = NodeUtil.isHoistedFunctionDeclaration(root);

      this.isFunction = root.getType() == Token.FUNCTION;

      if (root.getParent() != null) {
        int pType = root.getParent().getType();
        this.isLoop = pType == Token.DO ||
            pType == Token.WHILE ||
            pType == Token.FOR;
      } else {
        this.isLoop = false;
      }
    }