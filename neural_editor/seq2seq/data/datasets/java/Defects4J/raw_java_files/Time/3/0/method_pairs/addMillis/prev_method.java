
    public void addMillis(final int millis) {
            setMillis(getChronology().millis().add(getMillis(), millis));
    }