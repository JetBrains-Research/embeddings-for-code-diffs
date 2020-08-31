
    public Object getObject(Comparable rowKey, Comparable columnKey) {
        if (rowKey == null) {
            throw new IllegalArgumentException("Null 'rowKey' argument.");
        }
        if (columnKey == null) {
            throw new IllegalArgumentException("Null 'columnKey' argument.");
        }
        int row = this.rowKeys.indexOf(rowKey);
        if (row < 0) {
            throw new UnknownKeyException("Row key (" + rowKey 
                    + ") not recognised.");
        }
        int column = this.columnKeys.indexOf(columnKey);
        if (column < 0) {
            throw new UnknownKeyException("Column key (" + columnKey 
                    + ") not recognised.");
        }
        if (row >= 0) {
        KeyedObjects rowData = (KeyedObjects) this.rows.get(row);
            return rowData.getObject(columnKey);
        }
        else {
            return null;
        }
    }