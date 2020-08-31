
    public void removeObject(Comparable rowKey, Comparable columnKey) {
        setObject(null, rowKey, columnKey);
        
        // 1. check whether the row is now empty.
        boolean allNull = true;
        int rowIndex = getRowIndex(rowKey);
        KeyedObjects row = (KeyedObjects) this.rows.get(rowIndex);

        for (int item = 0, itemCount = row.getItemCount(); item < itemCount; 
             item++) {
            if (row.getObject(item) != null) {
                allNull = false;
                break;
            }
        }
        
        if (allNull) {
            this.rowKeys.remove(rowIndex);
            this.rows.remove(rowIndex);
        }
        
        // 2. check whether the column is now empty.
        allNull = true;
        
        for (int item = 0, itemCount = this.rows.size(); item < itemCount; 
             item++) {
            row = (KeyedObjects) this.rows.get(item);
            int columnIndex = row.getIndex(columnKey);
            if (columnIndex >= 0 && row.getObject(columnIndex) != null) {
                allNull = false;
                break;
            }
        }
        
        if (allNull) {
            for (int item = 0, itemCount = this.rows.size(); item < itemCount; 
                 item++) {
                row = (KeyedObjects) this.rows.get(item);
                int columnIndex = row.getIndex(columnKey);
                if (columnIndex >= 0) {
                    row.removeValue(columnIndex);
                }
            }
            this.columnKeys.remove(columnKey);
        }
    }