import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import jep.JepException;
import org.apache.uima.UIMAException;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CASRuntimeException;

import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.jcas.JCas;

import de.tudarmstadt.ukp.dkpro.core.corenlp.CoreNlpSegmenter;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class Main {
    public static void main(String[] args) throws UIMAException, CASRuntimeException, IOException {

		String[] tokens = "The old book is on the chair next to the table".split(" ");

        MiniExample mini = new MiniExample();
		mini.process(tokens);



        //AggregateBuilder builder = new AggregateBuilder();
        //builder.add(createEngineDescription(CoreNlpSegmenter.class));
        //builder.add(createEngineDescription(MiniExample.class));

        //SimplePipeline.runPipeline(jcas,builder.createAggregate());
        //System.out.println((jcas.getCas()));


        //writeXml2File(XmlFormatter.getPrettyString(jcas.getCas()), "D:/tea"+ ".xml");
        //writeJson2File(jcas, outputfolder + DocumentMetaData.get(jcas).getDocumentId() + ".json");




    }

    private static void writeXml2File(String file, String output) throws IOException {
        java.io.FileWriter fw = new java.io.FileWriter(output);
        fw.write(file);
        fw.close();
    }
}