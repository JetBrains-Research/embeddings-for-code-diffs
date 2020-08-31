
    public void add(DurationFieldType type, int amount) {
        if (type == null) {
            throw new IllegalArgumentException("Field must not be null");
        }
        if (amount != 0) {
            setMillis(type.getField(getChronology()).add(getMillis(), amount));
        }
    }