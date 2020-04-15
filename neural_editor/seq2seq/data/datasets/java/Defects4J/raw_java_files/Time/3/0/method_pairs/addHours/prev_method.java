
    public void addHours(final int hours) {
            setMillis(getChronology().hours().add(getMillis(), hours));
    }