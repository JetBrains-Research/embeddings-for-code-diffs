
    public void addWeeks(final int weeks) {
        if (weeks != 0) {
            setMillis(getChronology().weeks().add(getMillis(), weeks));
        }
    }