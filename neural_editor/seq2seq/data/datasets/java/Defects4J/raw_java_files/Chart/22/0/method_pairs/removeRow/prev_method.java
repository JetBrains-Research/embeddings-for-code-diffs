
    public void removeRow(Comparable rowKey) {
        int index = getRowIndex(rowKey);
        removeRow(index);
    }