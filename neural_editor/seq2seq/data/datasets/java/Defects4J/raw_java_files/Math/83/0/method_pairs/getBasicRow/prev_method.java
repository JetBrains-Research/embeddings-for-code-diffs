
    private Integer getBasicRow(final int col, boolean ignoreObjectiveRows) {
        Integer row = null;
        int start = getNumObjectiveFunctions();
        for (int i = start; i < getHeight(); i++) {
            if (MathUtils.equals(getEntry(i, col), 1.0, epsilon) && (row == null)) {
                row = i;
            } else if (!MathUtils.equals(getEntry(i, col), 0.0, epsilon)) {
                return null;
            }
        }
        return row;
    }