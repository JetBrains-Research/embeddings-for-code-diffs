
    public void addHours(final int hours) {
        if (hours != 0) {
            setMillis(getChronology().hours().add(getMillis(), hours));
        }
    }