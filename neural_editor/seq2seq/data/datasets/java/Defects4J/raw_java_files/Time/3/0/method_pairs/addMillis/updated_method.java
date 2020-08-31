
    public void addMillis(final int millis) {
        if (millis != 0) {
            setMillis(getChronology().millis().add(getMillis(), millis));
        }
    }