
    public void addSeconds(final int seconds) {
        if (seconds != 0) {
            setMillis(getChronology().seconds().add(getMillis(), seconds));
        }
    }