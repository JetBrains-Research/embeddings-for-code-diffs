
  public boolean recordBlockDescription(String description) {
    if (parseDocumentation) {
    populated = true;
    }
    return currentInfo.documentBlock(description);
  }