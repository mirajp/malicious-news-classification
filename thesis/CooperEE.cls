% Modified for Cooper Union and renamed to CooperEE 01-28-2010
% Revision: 07-25-2007
% Revision History
%   07-10-2008 : Corrected Alignment of signature boxes on Masters/PhD Approval page
%   07-25-2007 : Corrected some spelling errors
%   05-16-2006 : Added etd option and moved most packages from class file to template
%   05-15-2006 : Initial version.
%
% Known bugs
%   Having a tiny bit of the abstract spill to second page defeats page number removal.
%   Workaround: make the abstract a little longer or a little shorter.
%
% The BYUPhys class is for producing theses and dissertations 
% in the BYU department of physics and astronomy.  You can supply 
% the following optional arguments in the square brackets to 
% specify the thesis type:
%
%   honors  : Produces the honors thesis preliminary pages
%   masters : Produces the masters thesis preliminary pages
%   phd     : Produces the PhD dissertation preliminary pages
%
% The default format is appropriate for printing, with blank pages 
% inserted after the preliminary pages in twoside mode so you can 
% send it directly to a two-sided printer. However, for ETD 
% submission the blank pages need to be removed from the final output.
% The following option does this:
%
%   etd     : Produces an electronic copy with no blank pages in the preliminary section
%
% The rest of the class options are the same as the regular book class.  
% A few to remember:
%
%   oneside : Produces single sided print layout (recommended for theses less than 50 pages)
%   twoside : Produces single sided print layout (the default if you remove oneside)
%
% The BYUPhys class provides the following macros:
%
%   \makepreliminarypages : Makes the preliminary pages
%   \clearemptydoublepage : same as \cleardoublepage but doesn't put page numbers
%                           on blank intervening pages
%   \singlespace          : switch to single spaced lines
%   \doublespace          : switch to double spaced lines
%
% ------------------------------------------------------------------------------------------------------
%
\NeedsTeXFormat{LaTeX2e} \ProvidesClass{CooperEE}

% ---------------------------- declarations -------------------------
%
% These macros are used to declare arguments needed for the
% construction of the preliminary pages  

% The year and month the degree is awarded
\newcommand{\Year}[1]{\gdef\@Year{#1}}
\newcommand{\Month}[1]{\gdef\@Month{#1}}
\newcommand{\Day}[1]{\gdef\@Day{#1}}

% The full name of the degree
\newcommand{\degree}[1]{\gdef\@degree{#1}}

% The name of this document (thesis/dissertation)
\newcommand{\docname}[1]{\gdef\@docname{#1}}

% First line of title
\newcommand{\TitleTop}[1]{\gdef\@TitleTop{\mbox{\bf\LARGE{#1}}}}

% Second line of title
\newcommand{\TitleBottom}[1]{\gdef\@TitleBottom{\mbox{\bf\LARGE{#1}}}}

% Abstract text
\newcommand{\Abstract}[1]{\gdef\@Abstract{#1}}

% Acknowledgments text
\newcommand{\Acknowledgments}[1]{\gdef\@Acknowledgments{#1}}

% The author's name
\newcommand{\Author}[1]{\gdef\@Author{#1}}
\newcommand{\Cooper}[1]{\gdef\@Cooper{#1}}
\newcommand{\Nerken}[1]{\gdef\@Nerken{#1}}

% The name of the advisor
\newcommand{\Advisor}[1]{\gdef\@Advisor{#1}}

% The name of the dean
\newcommand{\Dean}[1]{\gdef\@Dean{#1}}

% The title of the dean
\newcommand{\DeanTitle}[1]{\gdef\@DeanTitle{#1}}

% The date of the thesis defense, for display next to signatures
\newcommand{\ThesisDefenseDate}[1]{\gdef\@ThesisDefenseDate{#1}}


% Set default values for fields
  \Year{1910}
  \Month{January}
  \Day{1}
  \Author{Author}
  \TitleTop{First line of title}
  \TitleBottom{ } % default is empty
  \Abstract{Abstract text goes here.}
  \Acknowledgments{Acknowledgment text goes here.}
  \degree{Master of Engineering}
  \docname{Master's Thesis}
  \Advisor{Dr. Advisor}
  \Dean{Dr. Dean}
  \DeanTitle{Dean, School of Engineering}

  \Cooper{THE COOPER UNION FOR THE ADVANCEMENT OF SCIENCE AND ART}
  \Nerken{ALBERT NERKEN SCHOOL OF ENGINEERING}
  
% ---------------------------- options ------------------------------

% A command to switch to single spaced lines
\newcommand{\singlespace}{\renewcommand{\baselinestretch}{1}\small\normalsize}

% A command to switch to double spaced lines
\newcommand{\doublespace}{\renewcommand{\baselinestretch}{1.66}\small\normalsize}

% A command pirated from chngpage.sty
\DeclareRobustCommand{\ch@ngetext}{%
  \setlength{\@colht}{\textheight}\setlength{\@colroom}{\textheight}%
  \setlength{\vsize}{\textheight}\setlength{\columnwidth}{\textwidth}%
  \if@twocolumn%
    \advance\columnwidth-\columnsep \divide\columnwidth\tw@%
    \@firstcolumntrue%
  \fi%
  \setlength{\hsize}{\columnwidth}%
  \setlength{\linewidth}{\hsize}%
}

% A command to make margins right for the initial single sided business.
\newcommand{\preliminarymargins}{%
    \addtolength{\textwidth}{-0.25in}%
    \addtolength{\evensidemargin}{0.25in}%
    \ch@ngetext%
    }

% A command to fix the margins after the initial single sided business.
\newcommand{\fixmargins}{%
    \addtolength{\textwidth}{0.25in}
    \addtolength{\evensidemargin}{-0.25in}
    \ch@ngetext%
}

% Define the preliminary section for a senior thesis.  
% The senior option is essentially ignored since it is the default 


% Changes to masters thesis preliminary section if the 'masters' option is specified
\DeclareOption{masters}{
  \degree{Master of Engineering}
  \docname{thesis}
  \newcommand{\makepreliminarypages}{
    \preliminarymargins
    \titlepage 
    \masterapprovalpage 
    \abstractpage 
    \acknowledgmentspage 
    \fixmargins
    \renewcommand{\clearemptydoublepage}{\cle@remptydoublep@ge}
  }
}


% --------------------- Some commands to handle the single sided preliminary pages ------------------

% Define the '\clearemptydoublepage' command to clear pages but not number any blank pages inserted.
% This is taken from the BYUThesis.cls file
\let\cle@rdoublep@ge\cleardoublepage
\newcommand{\cle@remptydoublep@ge}{
  \clearpage
  \if@twoside
  \ifodd\c@page\else
  \fi\fi
  {\pagestyle{empty}\cle@rdoublep@ge}}
\newcommand{\clearemptydoublepage}{\cle@remptydoublep@ge}


% Create an abstract environment which is single sided, even in a double sided book.
% again, this was taken from BYUThesis.cls
\def\skip@bstr@ctp@ges{\relax}
\def\@@skip@bstr@ctp@ges{%
  \if@twoside
   \ifodd\c@page\else
    \vbox{\vbox to \vsize{}}
    \clearpage\fi
   \else
  \fi
  \afterpage{\skip@bstr@ctp@ges}
}
\newenvironment{abstractenv}{
   \def\skip@bstr@ctp@ges{\@@skip@bstr@ctp@ges}
   \afterpage{\skip@bstr@ctp@ges \thispagestyle{empty}}
   \pagestyle{empty}
}

% Redefine above commands if etd option is specified.  The blank pages make printing nice,
% but they don't want them in the submitted PDF
%etd by default
\renewcommand{\clearemptydoublepage}{ \clearpage }
\renewenvironment{abstractenv}{\afterpage{\thispagestyle{empty}}\pagestyle{empty}}{} 

% ------------------------ Load the class and needed packages ---------------------------------

% Load the book class  
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}} 
\ProcessOptions\relax 
\LoadClass[letterpaper,12pt]{book}

% The afterpage package is required to make single sided formal pages 
% in a double sided environment
\RequirePackage{afterpage}

\RequirePackage{hyphenat}

% Note: the hyperref package is required to make an appropriate ETD.
% However, we don't require it here since it is supposed to be the last
% package loaded and students may want to load other packages in the 
% main tex file.  So that this class file doesn't crash if the student
% forgets to load hyperref, we have used the following commands below:
%
%   \providecommand\phantomsection{}
%   \providecommand\pdfbookmark[3][]{}
%
% These commands provide dummy versions of the macros, but won't
% bother the real versions if the hyperref package is loaded in the 
% tex file.


% ---------------------------- main code ----------------------------

% Set the margins to BYU specifications for the single sided preliminary
% pages.  At the end of the \makepreliminarypages command we fix the margins
% to be appropriate alternating values for double sided printing (if selected).
% If the \makepreliminarypages macro is not run, this never gets fixed. 
  \setlength{\marginparwidth}{0pt}
  \setlength{\marginparsep}{0pt}
  \setlength{\oddsidemargin}{0.5in}
  \setlength{\evensidemargin}{0in}
  \setlength{\textwidth}{6in}
  \setlength{\topmargin}{0in}
  \setlength{\leftmargin}{1.25in} %delete me?
  \setlength{\headheight}{0.125in}
  \setlength{\headsep}{0.25in}
  \setlength{\textheight}{8.6in}
  \setlength{\footskip}{0.25in}
  \raggedbottom

% Redefine the Table of Contents to deal with some blank page 
% and bookmarking issues relating to ETD submission
\let\TEMPtableofcontents\tableofcontents
\renewcommand{\tableofcontents}{
  \clearemptydoublepage
  \providecommand\phantomsection{} \phantomsection
  \addcontentsline{toc}{chapter}{Table of Contents}
  \TEMPtableofcontents
}
 
% Redefine the List of Figures to deal with some blank page 
% and bookmarking issues
\let\TEMPlistoffigures\listoffigures
\renewcommand{\listoffigures}{
  \providecommand\phantomsection{} \phantomsection
  \addcontentsline{toc}{chapter}{List of Figures}
  \TEMPlistoffigures
}

% Redefine the Bibliography to deal with a bookmarking issues
\let\TEMPbibliography\bibliography
\renewcommand{\bibliography}{
  \providecommand\phantomsection{} \phantomsection
  \addcontentsline{toc}{chapter}{Bibliography}
  \TEMPbibliography
}

%---------------------------- The Preliminary Page Definitions --------------------------
 
% ============================== Title Page ===============================
\renewcommand{\titlepage}{
    \thispagestyle{empty}
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Title Page}{bm:Title}    
    \vspace*{-0.5in}
    \@Cooper\\[1\baselineskip]
    \@Nerken\\[5\baselineskip]
    \@TitleTop\\[\baselineskip]
    \@TitleBottom\\[2\baselineskip]
    by\\[\baselineskip]
    \@Author
    \vfill
    A thesis submitted in partial fulfillment\\[\baselineskip]
    of the requirements for the degree of\\[\baselineskip]
    \@degree\\[\baselineskip]
    \vfill
    \@Month~\@Day,~\@Year \\[4\baselineskip]
    {\bf Advisor}\\[\baselineskip]
    \@Advisor
    \end{center}
    \vfill
    \clearemptydoublepage
  }


% =============================== Approval page =======================

\newcommand{\signatureboxL}[1]{
    \parbox[t]{3.6in}{
        \ \\[2\baselineskip]
        \rule{3.3in}{0.5pt}\\
        #1
    }
}

\newcommand{\signatureboxR}[1]{
   \begin{flushright}
    \parbox[t]{3.6in}{
        \begin{flushright}
        \rule{3.3in}{0.5pt}\\
        #1
        \end{flushright}
    }
   \end{flushright}
}

\newcommand{\masterapprovalpage}{
    \thispagestyle{empty}
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Approval}{bm:Approval}    
    \vspace*{0.375in}
    \@Cooper\\[1\baselineskip]
    \@Nerken\\[10\baselineskip]
    \end{center}
    \noindent
    \nohyphens{This thesis was prepared under the direction of the Candidate's Thesis Advisor and has received approval. It was submitted to the Dean of the School of Engineering and the full Faculty, and was approved as partial fulfillment of the requirements for the degree of Master of Engineering.}
    \\[3\baselineskip]
    %\signatureboxR{\@Dean\\\@DeanTitle}
    \signatureboxR{\@DeanTitle \hspace{1mm} - \hspace{1mm} \@ThesisDefenseDate}
    %\signatureboxL{\@Advisor\\Candidate's Thesis Advisor}\\
    \signatureboxL{\@Advisor \hspace{1mm} - \hspace{1mm} \@ThesisDefenseDate\\Candidate's Thesis Advisor}\\
    \vfill
    \clearemptydoublepage
  }

% ========================= Abstract ===================================

\newcommand{\abstractpage}{
    \thispagestyle{empty}
    %\begin{abstractenv}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Abstract}{bm:Abstract}    
    \@makechapterhead{Abstract}
    \renewcommand{\baselinestretch}{1.66}\small\normalsize
    %\noindent\@Abstract
	\@Abstract
    %\end{abstractenv}
    \clearemptydoublepage
  }

% ========================= Acknowledgments ==============================
\newcommand{\acknowledgmentspage}{
    \thispagestyle{empty}
    %\begin{abstractenv}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Acknowledgments}{bm:Acknowledgments}    
    \@makechapterhead{Acknowledgments}
    \renewcommand{\baselinestretch}{1.66}\small\normalsize
    \noindent\@Acknowledgments  
    %\end{abstractenv}
    \clearemptydoublepage
    
  }


