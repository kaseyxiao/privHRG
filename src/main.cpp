// ****************************************************************************************************
// *** COPYRIGHT NOTICE *******************************************************************************
// privHRG - fits a Hierarchical Random Graph (HRG) model to data under differential privacy
//
// This program is heavily based on Aaron Clauset's Hierarchical Random Graphs project
// (http://tuvalu.santafe.edu/%7Eaaronc/hierarchy/). All their programs are put online publicly and 
// an redistributed and modified under the terms of the GNU General Public License. Please see the detailed copyright claims in fitHRG.h
// and give credits to original authors of HRG if you use these programs.
//
// This program is freely distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY. 
// If you have any questions upon this program, please contact XIAO Qian (xiaoqiannus@gmail.com).
//
// 
//
// ****************************************************************************************************
// Author       : XIAO Qian  ( xiaoqiannus@gmail.com )
// Collaborators: CHEN Rui and TAN Kian-Lee
// Project      : Differentially Private Network Structural Inference
// Location     : National University of Singapore
// Created      : 14 Feb 2014
// Modified     : 9 Oct 2014    (cleaned up for public consumption)
//
// ****************************************************************************************************
//
// This program runs the MCMC with HRG model and the input graph G under calibrated distribution
//
// ****************************************************************************************************
// *** PROGRAM USAGE NOTES ****************************************************************************
// 
// The input graph file to the algorithm must be a text file containing an edge list for the graph in question; 
// nodes are indexed by integers only, indices are separated by a tab, and edges are terminated by a 
// carriage return. 
// For instance, here is a pair of triangles linked by a single edge:
//
// 1        2
// 1        3
// 2        3
// 4        5
// 4        6
// 5        6
// 1        4
//
// If the input .pairs file is formatted incorrectly, the program will crash.
//
// ****************************************************************************************************

#include <iostream>
#include <stdio.h>
#include <string>
#include "stdlib.h"
#include "time.h"
#include "dendro.h"
#include "graph.h"
#include "rbtree.h"
#include "MersenneTwister.h"
#include "filesystem.h"


using namespace std;
// ******** Structures and Constants **********************************************************************

struct ioparameters {
    int			n;				// number vertices in input graph
    int			m;				// number of edges in input graph

    string		d_dir;			// working directory
    string		f_in;			// name of input file (either .pairs or .hrg)
    bool		flag_f;			// flag for if -f invoked
    string		f_dg;			// name of output hrg file
    string		f_dg_info;		// name of output information-on-hrg file
    string      dg_scratch;
    string		f_stat;			// name of output statistics file
    string		f_pairs;			// name of output random graph file
    string		f_namesLUT;		// name of output names LUT file
    bool		flag_make;		// flag for if -make invoked
    string		s_scratch;		// filename sans extension
    string		s_tag;			// user defined filename tag
    string		start_time;		// time simulation was started
    int			timer;			// timer for reading input
    bool		flag_timer;		// flag for when timer fires
    bool		flag_compact;		// compact the Lxy file
    string      out_dir;        // output directory
    double      epsilon_hrg;    //noise scale for hrg
    double      epsilon_edge;   //noise scale for edge
    double      T;              //constant for differential privacy
    int         thresh_eq;      //threshold for manually forcing MCMC stop after thresh_eq*ioparm.n steps and reaching convergence
    int         thresh_stop;    //threshold for manually stop MCMC stop after thresh_stop*ioparm.n

};

// ******** Global Variables ******************************************************************************

ioparameters	ioparm;				// program parameters
rbtree		namesLUT;				// look-up table; translates input file node names to graph indices
dendro*		d;					// hrg data structure
unsigned int	t;					// number of time steps max = 2^32 ~ 4,000,000,000
double		bestL;				// best likelihood found so far
int			out_count;			// counts number of maximum found
unsigned int	period  = 10000;		// number of MCMC moves to do before writing stuff out; default: 10000
double*		Likeli;				// holds last k hrg likelihoods


// ******** Function Prototypes ***************************************************************************

string      num2str(const unsigned int);
bool		parseCommandLine(int argc, char * argv[]);
bool		readPairsFile();
void		recordHRG(const int, const int, const double);
void        recordHRGSample(const double, const int, const int);
void		recordNamesLUT();
bool        MCMCEquilibrium_Find(double);
bool        MCMCEquilibrium_Sample(int, double, bool);
void        recordLogL(double*, int);
bool        check_convergence(double *, int);
void        readNamesLUT();
void        recordRandomGraphStructure(const string , graph* );
void        recordRandomGraphInfo(const string, const double, const int);







// ******** Main Loop *************************************************************************************

int main(int argc, char * argv[]) {
    ioparm.n		= 0;					// DEFAULT VALUES for runtime parameters
    ioparm.timer   = 20;				//
    ioparm.s_tag   = "";				//

    time_t t1      = time(&t1);			//
    int num_samples    = 10;				// default value
    bool flag_control = true;


    if (parseCommandLine(argc, argv)) {
        d = new dendro;		// create hrg data structure



        ioparm.start_time = asctime(localtime(&t1));


        if (ioparm.flag_f) { 
            //1. read .pairs file, 
            //2. run MCMC until equilbrium
            //3. sample HRG and generate random graph

            readPairsFile();							// read input .pairs file
            //compute sensitivity
            if(ioparm.epsilon_hrg){                     //compute parameter T to adjust the underlying distribution
                double N = ceil(ioparm.n/2.0)* floor(ioparm.n/2.0);
                double p = 1.0/N;
                double deltaU = fabs(log(p)+(N-1)*log(1-p));
                ioparm.T = ioparm.epsilon_hrg/(2*deltaU);
                cout<<"epsilon hrg: "<<ioparm.epsilon_hrg<<endl;
                cout<<"T: "<<ioparm.T<<endl;
            }else{
                ioparm.T = 1.0;
                cout<<"T: "<<ioparm.T<<endl;
            }
	        cout<<"sensitivity for hrg: "<<ioparm.T<<endl;

            cout << ">> beginning convergence to equilibrium\n";
            if (!(MCMCEquilibrium_Find(ioparm.T)))   { return 0; }	// run it to equilibrium
            cout << "\n>> convergence critera met\n>> beginning sampling\n";
            if (!(MCMCEquilibrium_Sample(num_samples, ioparm.T, flag_control))) { return 0; }	// sample likelihoods for missing connections
            cout << ">> sampling finished" << endl;
        } else if (ioparm.flag_make) {          //make random graph directly from .hrg dendrogram file
            readNamesLUT();
            cout << ">> begin import dendrogram: "<<ioparm.f_dg<<endl;
            cout << ">> make random graph with epsilon value: "<<ioparm.epsilon_edge<<endl;
            if (!(d->importDendrogramStructure(ioparm.f_dg))) { cout << "Error: Malformed input file.\n"; return 0; }

            int num_of_graph = 1;
            int i=0;

            while(i<num_of_graph){
                i++;
                graph* randomG = new graph(ioparm.n);
                char timestamp[30]  =  "";
                filesystem::getTimeStamp(timestamp);
                string outfile=ioparm.out_dir +  ioparm.dg_scratch+"_random_graph_"+timestamp;
                double thisL = d->getLikelihood();


                d->makeNoisyRandomGraph(randomG, ioparm.epsilon_edge, flag_control); //use Erdős–Rényi random graph model if flag_control is true
                recordRandomGraphStructure(outfile+".pairs", randomG);
                cout<<"number of nodes: "<<randomG->numNodes()<<endl;
                cout<<"number of links: "<<randomG->numLinks()<<endl;
                cout<<"number of simple edges: "<<randomG->numLinks()/2<<endl;
                int num_of_edges = randomG->numLinks()/2;
                recordRandomGraphInfo(outfile+".info", thisL, num_of_edges);



            }
        }

        return 1;
    } else { return 0; }

}


// ******** Function Definitions **************************************************************************

void recordRandomGraphInfo(const string outfile, const double thisL, const int m){
    // write statistics about random graph to file

    ofstream fout(outfile.c_str(), ios::trunc);
    fout << "---HIERARCHICAL-RANDOM-GRAPH---\n";
    fout << "StartTime     : " << ioparm.start_time;
    fout << "Directory     : " << ioparm.out_dir   << "\n";
    fout << "---Basic-Graph-Information---\n";
    fout << "Nodes         : " << ioparm.n		<< "\n";
    fout << "Edges         : " << m << "\n";
    fout << "---HRG-Information---\n";
    fout << "HRG           : " << ioparm.f_dg << "\n";
    fout << "logL          : " << thisL<<"\n";
    fout << "epsilon_edge  : " << ioparm.epsilon_edge<<"\n";
    fout.close();
}


// ********************************************************************************************************
void recordRandomGraphStructure(const string out_file, graph* random_g) {
    // write random graph to file
    edge* curr;
    string thisName;
    bool flag_debug = true;
    if (flag_debug) { cout << ">> dendro: writing random graph to file" << endl; }

    ofstream fout(out_file.c_str(), ios::trunc);
    for (int i=0; i<ioparm.n; i++) {
        curr     = random_g->getNeighborList(i);
        thisName = num2str(namesLUT.returnValue(i));      //get name from original graph
        while (curr != NULL) {
            if (thisName=="") { fout << i << "\t" << curr->x << "\n"; }
            else {              fout << thisName << "\t" << num2str(namesLUT.returnValue(curr->x))<< "\n"; }
            curr = curr->next;
        }
    }
    fout.close();

    return;
}

// ********************************************************************************************************

bool parseCommandLine(int argc, char * argv[]) {
    //parse command line parameters
    int argct = 1;
    string temp, ext;
    string::size_type pos;
    bool safeExit = false;

    if (argc==1) {
        cout << "\n  -- Differentially Private Network Structural Inference--\n";
        cout << "  privHRG is a command line program that takes a simple graph file and runs\n";
        cout << "  a Markov chain Monte Carlo algorithm to sample Hierarchical Random Graph models\n";
        cout << "  from a calibrated distribution in order to satisfying differential privacy.\n";
        cout << "  -f <file>       Input .pairs graph file\n";
        cout << "  -epsilonHRG <real number>    Input privacy budget for HRG\n";
        cout << "  -epsilonE <real number>  Input privacy budget for edge perturbation\n";
        cout << "  -make <file>    Build random graph from <dataname>_<tag>-dendro.hrg dendrogram file\n";
        cout << "  -eq <integer>  threshold for manually forcing MCMC stop after eq*n steps and reaching convergence\n";
        cout << "  -stop <integer> threshold for manually stop MCMC stop after stop*n\n";
        cout << "\n";
        cout << "  ./privHRG -f data/karate.pairs\n";
        cout << "  ./privHRG -f data/karate.pairs -epsilonHRG 0.5 -epsilonE 0.5\n";
        cout << "  ./privHRG -f data/karate.pairs -epsilonHRG 0.5 -epsilonE 0.5 -eq 3000 -stop 4000\n";
        cout << "  ./privHRG -make karate_sample-dendrogram.hrg -epsilonE 0.5\n";
        cout << "\n";
        return false;

    } else {
        while (argct < argc) {
            temp = argv[argct];

            if (temp == "-make" and !ioparm.flag_f) {
                ioparm.flag_make = true;				// -make is mutually exclusive with -f
                argct++;
                temp = argv[argct];
                ext = ".hrg";
                pos = temp.find(ext,0);
                if (pos == string::npos) { cout << " Error: Input file must claim to be .hrg format.\n"; return safeExit; }
                ioparm.f_in = ioparm.f_dg = temp;
                ext = "/";
                pos = string::npos;
                for (int i=0; i < temp.size(); i++) { if (temp[i] == '/') { pos = i; } }
                if (pos != string::npos) {
                    ioparm.d_dir = temp.substr(0, pos+1);
                    temp = temp.substr(pos+1,temp.size()-pos-1);
                }else{
                    ioparm.d_dir = "";
                }
                // now grab the filename sans extension for building outputs files
                for (int i=0; i< temp.size(); i++){if (temp[i] == '_'){pos = i; break;}}
                ioparm.f_namesLUT = "data/" + temp.substr(0, pos) + "-names.lut";
//                for (int i=0; i < temp.size(); i++) { if (temp[i] == '.') { pos = i; } }
                ioparm.s_scratch = temp.substr(0,pos);
                safeExit         = true;

            } else if (temp == "-t")       { argct++; ioparm.s_tag = argv[argct];
            } else if (temp == "-compact") { ioparm.flag_compact = true;
            } else if(temp == "-epsilonHRG") {argct++; ioparm.epsilon_hrg = strtod(argv[argct], NULL) ;
            } else if (temp == "-epsilonE"){argct++; ioparm.epsilon_edge = strtod(argv[argct], NULL);
            } else if(temp == "-eq") {argct++; ioparm.thresh_eq = int(strtod(argv[argct], NULL)) ;
            } else if(temp == "-stop") {argct++; ioparm.thresh_stop = int(strtod(argv[argct], NULL)) ;

            } else if (temp == "-f" and !ioparm.flag_make) {
                ioparm.flag_f = true;				// -f is mutually exclusive with -make
                argct++;
                temp = argv[argct];
                ext = ".pairs";
                pos = temp.find(ext,0);
                if (pos == string::npos) { cout << " Error: Input file must claim to be .pairs format.\n"; return safeExit; }
                ioparm.f_in = temp;
                ext = "/";
                pos = string::npos;
                for (int i=0; i < temp.size(); i++) { if (temp[i] == '/') { pos = i; } }
                if (pos != string::npos) {
                    ioparm.d_dir = temp.substr(0, pos+1);
                    temp = temp.substr(pos+1,temp.size()-pos-1);
                }
                // now grab the filename sans extension for building outputs files
                for (int i=0; i < temp.size(); i++) { if (temp[i] == '.') { pos = i; } }
                ioparm.s_scratch = temp.substr(0,pos);
//				ioparm.f_stat    = ioparm.d_dir + ioparm.s_scratch + "-L.xy";
                safeExit         = true;

            } else { cout << " Warning: ignored argument " << argct << " : " << temp << endl; }
            argct++;
        }
    }
    if(!ioparm.flag_make){ioparm.f_namesLUT = ioparm.d_dir + ioparm.s_scratch + "-names.lut";}
    if (ioparm.s_tag != "")    { ioparm.s_scratch += "_" + ioparm.s_tag; }
    if (ioparm.flag_make)      { ioparm.f_pairs    = ioparm.d_dir + ioparm.s_scratch + "-random.pairs"; }
    if (ioparm.flag_f)         { ioparm.f_stat     = ioparm.d_dir + ioparm.s_scratch + "-L.xy";  }
    filesystem::create_directory("output");
    if(!ioparm.flag_make){
        char timestampDir[30]  =  "output/";
        filesystem::timeStampDir(timestampDir);
        ext = "/";
        ioparm.out_dir = timestampDir + ext;

        filesystem::create_directory(ioparm.out_dir.c_str());

    }else{
        string::size_type pos1, pos2;
        pos1 = -1;
        pos2 = string::npos;
        for (int i=0; i < ioparm.f_dg.size(); i++) {
            if (ioparm.f_dg[i] == '/') { pos1 = i; }
            if (ioparm.f_dg[i] == '.') { pos2 = i; }
        }




        ioparm.dg_scratch = ioparm.f_dg.substr(pos1+1, pos2-(pos1+1));


        ioparm.out_dir =  ioparm.d_dir;
        cout<<ioparm.dg_scratch<<endl;
    }
    return safeExit;
}
// ********************************************************************************************************
bool readPairsFile() {

    int n,m,s,f,a,b;    n = m = 0;
    elementrb *item;
    time_t t1; t1 = time(&t1);
    time_t t2; t2 = time(&t2);

    // First, we scan through the input file to create a list of unique node names
    // (which we store in the namesLUT), and a count of the number of edges.
    cout << ">> input file scan ( " << ioparm.f_in << " )" << endl;
    cout << "   edges: [0]"<<endl;
    ifstream fscan1(ioparm.f_in.c_str(), ios::in);
    while (fscan1 >> s >> f) {					// read friendship pair (s,f)
        if (s != f) {
            m++;								// count number of edges
            if (namesLUT.findItem(s) == NULL) { namesLUT.insertItem(s, n++); }
            if (namesLUT.findItem(f) == NULL) { namesLUT.insertItem(f, n++); }
        }

        if (t2-t1>ioparm.timer) {				// check timer; if necessarsy, display
            cout << "   edges: ["<<m<<"]"<<endl;
            t1 = t2; ioparm.flag_timer = true;		//
        }									//
        t2=time(&t2);							//
    }
    fscan1.close();
    cout << "   edges: ["<<m<<"]"<<endl;
    d->g = new graph (n);						// make new graph with n vertices

    // Finally, we reparse the file and added edges to the graph
    m = 0;
    ioparm.flag_timer = false;					// reset timer

    cout << ">> input file read ( " << ioparm.f_in << " )" << endl;
    cout << "   edges: [0]"<<endl;
    ifstream fin(ioparm.f_in.c_str(), ios::in);
    while (fin >> s >> f) {
        m++;
        if (s != f) {
            item = namesLUT.findItem(s); a = item->value;
            item = namesLUT.findItem(f); b = item->value;
            if (!(d->g->doesLinkExist(a,b))) { if (!(d->g->addLink(a,b))) { cout << "Error: (" << s << " " << f << ")" << endl; } else if (d->g->getName(a) == "") { d->g->setName(a, num2str(s)); } }
            if (!(d->g->doesLinkExist(b,a))) { if (!(d->g->addLink(b,a))) { cout << "Error: (" << s << " " << f << ")" << endl; } else if (d->g->getName(b) == "") { d->g->setName(b, num2str(f)); } }
        }
        if (t2-t1>ioparm.timer) {				// check timer; if necessarsy, display
            cout << "   edges: ["<<m<<"]"<<endl;
            t1 = t2; ioparm.flag_timer = true;		//
        }									//
        t2=time(&t2);							//
    }
    cout << ">> edges: ["<<m<<"]"<<endl;
    fin.close();
    ioparm.m = d->g->numLinks();					// store actual number of directional edges created
    ioparm.n = d->g->numNodes();					// store actual number of nodes used
    cout << "vertices: ["<<ioparm.n<<"]"<<endl;

    recordNamesLUT();							// record names LUT to file for future reference
    cout << ">> recorded names look-up table" << endl;

    d->buildDendrogram();
    cout<<"number of nodes: "<<d->g->numNodes()<<endl;
    cout<<"number of links: "<<d->g->numLinks()<<endl;

    return true;
}
// ********************************************************************************************************
bool MCMCEquilibrium_Find(double T) {
    double	dL, Likeli, bestL, oldMeanL, newMeanL;
    bool		flag_taken, flag_eq;
    int       t = 1;
    int     thresh_eq, thresh_stop;
    //set two threshold parameters to manually control convergence
    if(ioparm.thresh_eq){
        thresh_eq = max(ioparm.thresh_eq, 1000);
    }else{
        thresh_eq = 1000;
    }
    if(ioparm.thresh_stop){
        thresh_stop = max(ioparm.thresh_stop, 3000);
    }else{
        thresh_stop = 3000;
    }


    // We want to run the MCMC until we've found equilibrium; we use the heuristic of the
    // average log-likelihood (which is exactly the entropy) over X steps being very close
    // to the average log-likelihood (entropy) over the X steps that preceded those. In other
    // words, we look for an apparent local convergence of the entropy measure of the MCMC.

    cout << "\nstep   \tLogL       \tbest LogL\tMC step\n";
    newMeanL = -1e49;
    int interval = 65536;
    int max_len = max(int(1e8), thresh_eq*ioparm.n);
    double * trace = new double[max_len];
    int len =0;
    double * check = new double[10000];
    int check_num = 0;


    flag_eq = false;
    bestL = d->getLikelihood();
    while (!flag_eq) {
        oldMeanL = newMeanL;
        newMeanL = 0.0;

        for (int i=0; i<interval; i++) {


            if (!(d->monteCarloMove(dL, flag_taken, T))) {	// Make a single MCMC move
                return false; }

            Likeli = d->getLikelihood();			// get likelihood of this D
            if (Likeli > bestL) { bestL = Likeli; }	// store the current best likelihood
            newMeanL += Likeli;

            if(len < max_len){
                   trace[len]=Likeli;
                   len++;
            }




            // Write some stuff to standard-out to describe the current state of things.
            if (t % 16384 == 1) {
                cout << "[" << t << "]- \t " << Likeli << " \t(" << bestL << ")\t";
                if (flag_taken) { cout << "*\t"; } else { cout << " \t"; }
                cout << endl;

            }

            if (t > 2147483640 or t < 0) { t = 1; } else { t++; }
        }
        d->refreshLikelihood();					// correct floating-point errors O(n)

        // Check if localized entropy appears to have stabilized; if you want to use a
        // different convergence criteria, this is where you would change things.
//        if (fabs(newMeanL - oldMeanL)/interval < 1.0 and (t>10000*ioparm.n )) { flag_eq = true; }
        check[check_num] = fabs(newMeanL - oldMeanL)/interval;
        cout<<"newMeanL-oldMeanL: "<< check[check_num]<<endl;
        check_num++;
        if(check_convergence(check, check_num) and (t>thresh_eq*ioparm.n)){

            flag_eq = true;
        }else if(t>thresh_stop*ioparm.n or t>=max_len){
            flag_eq = true;
        }




    }
    recordLogL(trace, len);
    delete [] trace;
    trace		= NULL;
    delete [] check;
    check = NULL;

    return true;
}

bool check_convergence(double *check_trace, int len){
    int stationary =0;
    double mean = 0;
    int thresh = 5;
    int non_stationary = 0;
    if(len>=thresh){
        for(int i=1; i<=thresh; i++){
            if(check_trace[len-i]<0.02*ioparm.n)
                stationary++;
            if(check_trace[len-i]>0.05*ioparm.n){
                non_stationary++;
            }

            mean += check_trace[len-i]/thresh;
        }
        if(non_stationary<=0 and stationary>=(int)(thresh*0.8)){
            return true;
        }else{
            return false;
        }

    }else{
        return false;
    }

}

void recordLogL(double*trace, int len){
    string outfile;
    outfile = ioparm.out_dir+ioparm.s_scratch+"-logL-trace.txt";
    ofstream fout(outfile.c_str(), ios::trunc);
    fout << "#step\tlogL\n";
    int i=0;
    int interval = ioparm.n;

    while(i<len) {
        fout << i << "\t" << trace[i] << "\n";
        i +=  interval;

    }
    fout.close();

    return;
}



// ********************************************************************************************************

bool MCMCEquilibrium_Sample(int num_samples, double T, bool flag_control=true) {
    double	dL, Likeli, bestL;
    bool		flag_taken;
//    double	ptest       = 1.0/10.0;
    double ptest = (4.0/(double)(ioparm.n));

    int		thresh      = 100*ioparm.n;
    int		t           = 1;
    int		sample_num  = 0;
    MTRand    mtr;

    // Because moves in the dendrogram space are chosen (Monte Carlo) so that we sample dendrograms
    // with probability proportional to their likelihood, a likelihood-proportional sampling of
    // the dendrogram models would be equivalent to a uniform sampling of the walk itself. We would
    // still have to decide how often to sample the walk (at most once every n steps is recommended)
    // but for simplicity, the code here simply runs the MCMC itself. To actually compute something
    // over the set of sampled dendrogram models (in a Bayesian model averaging sense), you'll need
    // to code that yourself.

    cout << "\nstep   \tLogL       \tbest LogL\tMC step\t% complete\n";
    bestL = d->getLikelihood();
    while (sample_num < num_samples) {
        for (int i=0; i<65536; i++) {
            // Make a single MCMC move
            if (!(d->monteCarloMove(dL, flag_taken, T))) { return false; }
            Likeli = d->getLikelihood();				// get this likelihood
            if (Likeli > bestL) { bestL = Likeli; }		// check if this logL beats best

            // We sample the dendrogram space every 1/ptest MCMC moves (on average).
            if (t > thresh and mtr.randExc() < ptest) {
                sample_num++;
                recordHRGSample(Likeli, sample_num, t);
                graph* randomG = new graph(ioparm.n);
                d->makeNoisyRandomGraph(randomG, ioparm.epsilon_edge, flag_control);
                cout << "begin making random graph with dendrogram of likelihood: "<<Likeli<<endl;
                cout << "epsilon for hrg: "<<ioparm.epsilon_hrg<<endl;
                cout << "epsilon for edge: "<<ioparm.epsilon_edge<<endl;
                string outfile=ioparm.out_dir + ioparm.s_scratch + "_random_graph_"+ num2str(sample_num);
                recordRandomGraphInfo(outfile+".info", Likeli, randomG->numLinks()/2);
                d->recordGraphStructure(outfile+".pairs", randomG);
                cout << "["<<t<<"]-\t"<<"get"<<sample_num << " sample, logL: " <<Likeli<<endl;

                if (sample_num >= num_samples) { i = 65536; }

            }

            // Write some stuff to standard-out to describe the current state of things.
            if (t % 16384 == 1) {
                cout << "[" << t << "]+ \t " << Likeli << " \t(" << bestL << ")\t";
                if (flag_taken) { cout << "*\t"; } else { cout << " \t"; }
                cout << (double)(sample_num)/(double)(num_samples) << endl;
            }

            if (t > 2147483640 or t < 0) { t = 1; } else { t++; }
        }
        d->refreshLikelihood();						// correct floating-point errors O(n)
    }

    return true;
}

// ********************************************************************************************************

string num2str(const unsigned int input) {
    // input must be a positive integer
    unsigned int temp = input;
    string str  = "";
    if (input == 0) { str = "0"; } else {
        while (temp != 0) {
            str  = char(int(temp % 10)+48) + str;
            temp = (unsigned int)temp/10;
        }
    }
    return str;
}

// ********************************************************************************************************

void recordHRGSample(const double thisL, const int sample_num, const int step){
    time_t t1;

    // write hrg to file
    ioparm.f_dg = ioparm.out_dir + ioparm.s_scratch + "_sample_"+ num2str(sample_num)+"-dendro.hrg";
    d->recordDendrogramStructure(ioparm.f_dg);

    // write statistics about hrg to file
    ioparm.f_dg_info = ioparm.out_dir + ioparm.s_scratch + "_sample_"+num2str(sample_num)+"-dendro.info";

    t1 = time(&t1);
    ofstream fout(ioparm.f_dg_info.c_str(), ios::trunc);
    fout << "---HIERARCHICAL-RANDOM-GRAPH---\n";
    fout << "StartTime     : " << ioparm.start_time;
    fout << "InputFile     : " << ioparm.f_in    << "\n";
    fout << "Directory     : " << ioparm.d_dir   << "\n";
    fout << "---Basic-Graph-Information---\n";
    fout << "Nodes         : " << ioparm.n		<< "\n";
    fout << "Edges         : " << ioparm.m/2	<< "\n";
    fout << "---HRG-Information---\n";
    fout << "OutputTime    : " << asctime(localtime(&t1));
    fout << "NumStepsMCMC  : " << step			<< "\n";
//    fout << "NumPrevBests  : " << count-1		<< "\n";
    fout << "LogLikelihood : " << thisL		<< "\n";
    fout << "epsilon_hrg : " << ioparm.epsilon_hrg		<< "\n";
    fout << "epsilon_edge : " << ioparm.epsilon_edge<<"\n";
    fout << "T:             " << ioparm.T <<"\n";
    fout << "HRG           : " << ioparm.f_dg << "\n";
    fout << "InfoFile      : " << ioparm.f_dg_info  << "\n";
    fout.close();

    return;
}

// ********************************************************************************************************

void recordHRG(const int step, const int count, const double thisL) {

    time_t t1;

    // write hrg to file
    ioparm.f_dg = ioparm.d_dir + ioparm.s_scratch + "_best-dendro.hrg";
    d->recordDendrogramStructure(ioparm.f_dg);

    // write statistics about hrg to file
    ioparm.f_dg_info = ioparm.d_dir + ioparm.s_scratch + "_best-dendro.info";

    t1 = time(&t1);
    ofstream fout(ioparm.f_dg_info.c_str(), ios::trunc);
    fout << "---HIERARCHICAL-RANDOM-GRAPH---\n";
    fout << "StartTime     : " << ioparm.start_time;
    fout << "InputFile     : " << ioparm.f_in    << "\n";
    fout << "Directory     : " << ioparm.d_dir   << "\n";
    fout << "---Basic-Graph-Information---\n";
    fout << "Nodes         : " << ioparm.n		<< "\n";
    fout << "Edges         : " << ioparm.m/2	<< "\n";
    fout << "---HRG-Information---\n";
    fout << "OutputTime    : " << asctime(localtime(&t1));
    fout << "NumStepsMCMC  : " << step			<< "\n";
    fout << "NumPrevBests  : " << count-1		<< "\n";
    fout << "LogLikelihood : " << thisL		<< "\n";
    fout << "HRG           : " << ioparm.s_scratch + "_best-dendro.hrg" << "\n";
    fout << "InfoFile      : " << ioparm.s_scratch + "_best-dendro.info"  << "\n";
    fout.close();

    return;
}

// ********************************************************************************************************
void readNamesLUT(){

    int n,s, f;
    n=0;
    string headline;
    ifstream fscan1(ioparm.f_namesLUT.c_str(), ios::in);
    getline(fscan1, headline);
    while (fscan1 >> s >> f) {					// read friendship pair (s,f)
            if (namesLUT.findItem(s) == NULL) {
                namesLUT.insertItem(s, f);
                n++;

        }

    }
    fscan1.close();
    ioparm.n = n;

    return;

}
void recordNamesLUT() {
    rbtree reverseNamesLUT;
    keyValuePair *head, *prev;

    head = namesLUT.returnTreeAsList();
    while (head != NULL) {
        reverseNamesLUT.insertItem(head->y, head->x);
        prev = head;
        head = head->next;
        delete prev;
    }
    head = NULL; prev = NULL;

    elementrb *item;
    ofstream fout(ioparm.f_namesLUT.c_str(), ios::trunc);
    fout << "virtual\treal\n";
    for (int i=0; i<ioparm.n; i++) {
        item = reverseNamesLUT.findItem(i);
        fout << i << "\t" << item->value << "\n";
    }
    fout.close();

    return;
}

// ********************************************************************************************************
// ********************************************************************************************************


