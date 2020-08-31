
  private void unrollBinaryOperator(
      Node n, int op, String opStr, Context context,
      Context rhsContext, int leftPrecedence, int rightPrecedence) {
    Node firstNonOperator = n.getFirstChild();
    while (firstNonOperator.getType() == op) {
      firstNonOperator = firstNonOperator.getFirstChild();
    }

    addExpr(firstNonOperator, leftPrecedence, context);

    Node current = firstNonOperator;
    do {
      current = current.getParent();
      cc.listSeparator();
      addExpr(current.getFirstChild().getNext(), rightPrecedence, rhsContext);
    } while (current != n);
  }