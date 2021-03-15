package com.example.helloworld;
import java.io.*;

public class HelloWorld {
    public static void main(String[] args) throws Exception {
        String pythonScriptPath = "C:/Users/Kristina/IdeaProjects/HelloWorld/hello.py";
        String[] cmd = new String[2];
        cmd[0] = "python"; // check version of installed python: python -V
        cmd[1] = pythonScriptPath;

        Runtime rt = Runtime.getRuntime();
        Process pr = rt.exec(cmd);

        BufferedReader bfr = new BufferedReader(new InputStreamReader(pr.getInputStream()));
        String line = "";
        while ((line = bfr.readLine()) != null) {
            System.out.println(line);
        }
    }
}
