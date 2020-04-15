
    public void addMinutes(final int minutes) {
            setMillis(getChronology().minutes().add(getMillis(), minutes));
    }