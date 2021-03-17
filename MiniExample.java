import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import org.apache.uima.UIMAException;
import org.apache.uima.UimaContext;
import org.apache.uima.cas.CASRuntimeException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.resource.ResourceInitializationException;
import org.texttechnologylab.annotation.semaf.isobase.Entity;
import org.texttechnologylab.annotation.semaf.isospace.QsLink;
import org.texttechnologylab.annotation.semaf.isospace.SpatialEntity;
import org.texttechnologylab.annotation.type.QuickTreeNode;

import java.io.IOException;
import java.io.*;
import java.util.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.StringTokenizer;

import static org.apache.uima.fit.util.JCasUtil.select;
import static org.apache.uima.fit.util.JCasUtil.selectCovered;

public class MiniExample {
    List<String> words;

    public void process(String[] tokens) {
        List<String> words = Arrays.asList("cabinet", "objects", "shower", "bathtub", "wall",
                "window", "ceiling", "towel", "counter", "lighting", "door", "mirror", "curtain", "sink",
                "floor", "picture", "toilet", "chair", "bed", "chest_of_drawers", "cushion", "stool", "void",
                "table", "tv_monitor", "plant", "shelving", "appliances", "misc", "sofa", "fireplace", "column",
                "beam", "railing", "stairs", "seating", "clothes", "furniture");
        List<String> listWithImportantWords = new ArrayList<>();
        List<SpatialEntity> entityList = new ArrayList<>();
        System.out.println("ArrayList: " + words);
        for (String token : tokens) {
            if (words.contains((token))) {
                listWithImportantWords.add(token);
            }
        }
        System.out.println(listWithImportantWords);
    }
}


        /*for (Token token : select(aJCas, Token.class)) {
            if (words.contains(token.getCoveredText())) {
                //Eingabe Interpreter
                listWithImportantWords.add(token.getCoveredText());

                //Wichtig für TextImager
                SpatialEntity entity = new SpatialEntity(aJCas);
                entity.setBegin(token.getBegin());
                entity.setEnd(token.getEnd());
                entity.addToIndexes();
                entityList.add(entity);
            }
        }
        System.out.println(listWithImportantWords);
        //Give interpreter

        int token1 = 0;
        int token2 = 1;
        String rel = "IN";
        QsLink link = new QsLink(aJCas);
        link.setFigure(entityList.get(token1));
        link.setGround(entityList.get(token2));
        link.setRel_type(rel);
        link.addToIndexes();
    }
}*/