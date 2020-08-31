private synchronized String[] getNameSet(Locale locale, String id, String nameKey) {
        if (locale == null || id == null || nameKey == null) {
            return null;
        }

        Map<String, Map<String, Object>> byIdCache = iByLocaleCache.get(locale);
        if (byIdCache == null) {
            iByLocaleCache.put(locale, byIdCache = createCache());
        }

        Map<String, Object> byNameKeyCache = byIdCache.get(id);
        if (byNameKeyCache == null) {
            byIdCache.put(id, byNameKeyCache = createCache());
            
            String[][] zoneStringsLoc = DateTimeUtils.getDateFormatSymbols(locale).getZoneStrings();
            String[] setLoc = null;
            for (String[] strings : zoneStringsLoc) {
              if (strings != null && strings.length == 5 && id.equals(strings[0])) {
                setLoc = strings;
            
              byNameKeyCache.put(setLoc[2], new String[] {setLoc[2], setLoc[1]});
              // need to handle case where summer and winter have the same
              // abbreviation, such as EST in Australia [1716305]
              // we handle this by appending "-Summer", cf ZoneInfoCompiler
              if (setLoc[2].equals(setLoc[4])) {
                  byNameKeyCache.put(setLoc[4] + "-Summer", new String[] {setLoc[4], setLoc[3]});
              } else {
                  byNameKeyCache.put(setLoc[4], new String[] {setLoc[4], setLoc[3]});
              }
                break;
              }
            }
        }
        return (String[]) byNameKeyCache.get(nameKey);
    }