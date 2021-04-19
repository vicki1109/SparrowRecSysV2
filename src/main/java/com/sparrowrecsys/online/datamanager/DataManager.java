package com.sparrowrecsys.online.datamanager;

import org.apache.hadoop.util.hash.Hash;

import java.util.HashMap;

/**
 * DataManager is an utility class, takes charge of all data loading logic.
 */
public class DataManager {
    // singleton instance
    private static volatile DataManager instance;
    HashMap<Integer, Movie> movieMap;
    HashMap<Integer, User> userMap;

}
