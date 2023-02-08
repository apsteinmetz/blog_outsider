---
title: Why I migrated from Excel to R
author: Art Steinmetz
date: '2019-06-12'
slug: why-i-migrated-from-excel-to-r
categories:
  - R
  - Excel
tags: []
description: ''
topics: []
---
![Old Spreadsheets](img/spreadsheet_mosaic.png)

I’ve been a spreadsheet power user from the days of Visicalc for the Apple ][.  I migrated to Lotus 1-2-3, to Borland Quattro and finally to Excel.  With Excel, I’ve bludgeoned Visual Basic to create some pretty complicated dashboards and analytics.  When I started using R I used tools like [RExcel](http://rcom.univie.ac.at/) that plug R into Excel as an analytic server, or I would use Excel to download data from investment databases and export it for use in R.  But now I find I open up Excel only rarely and do all my quantitative investigations entirely within R.  Why?

1. Simplicity.  R uses is a very different paradigm than a spreadsheet so it takes some getting used to.  On the surface, R is a programming language, like C or Java.  Spreadsheets were invented to free humans to get real analytic work done without becoming coders.  Yet R comes at the coding angle from a very different direction than application languages.  It is a data science tool first and a programming language second.  Like a spreadsheet, the central unit of analytic work is data in a row and column format.  Once you master the vernacular of manipulating these “data frames” the rest is easy.  Like a spreadsheet, R is interactive.  You try single operations and can paste the successful ones into your “program” in a sequential fashion, building up your analysis step by step.  In doing so you are creating a log of your work that lets you pick up later where you left off without missing a beat or reuse bits in other projects. Most people use the [R Studio development environment](https://www.rstudio.com/products/rstudio/) as their workbench. It's simple, powerful and free!

2. Auditability.  Big Excel spreadsheets are a labyrinth of linked formulas.  Tracing errors is extremely difficult.  Noticing errors at all is often tough.  They often go unnoticed for the life of the spreadsheet.  When I come back to a spreadsheet I’ve created months ago, I often can’t remember how I did something. If something breaks, tracking down the broken piece takes a long time.  So while spreadsheets are easy to master, mastery comes at the expense of maintainability.  With R, each step of the analysis proceeds in a roughly linear fashion.  Each piece building on the previous one.  It is easy to see where the problems are and to insert the fixes without blowing up something else you didn’t realize was connected.

3. Reproducibility.  How often do you share a big spreadsheet with someone else?  Can they use it? Have you been bequeathed a spreadsheet that is part of the team workflow that you have no idea how to maintain?  With R the logic of the analysis flows in discrete steps so every step is immediately visible.  The code is its own log of all the work performed.  That’s not to say you are off the hook for documentation.  Well commented code is a sign of disciplined thinking and a courtesy to both others and your future self.  Inserting a comment line in R code comes naturally, I find, while it requires conscious effort in Excel.

4. Shareability.  “Notebooks” are the new thing and I love them.  These are HTML documents, [like this blog](https://outsiderdata.netlify.com/), that include descriptive text, R code (or whatever language you use) and the output of the code.  They make sharing and showing off your work visually attractive and simple to follow. You can attach the HTML document to an email, render it as a PDF or publish it to a web site.  Embedding markup language is a little extra work but the R Studio IDE creates the templates automatically, and the result is sharp.

5. Visualization.  It is easy to create charts in Excel.  It is easy to create charts in R, though it is done in code, not interactively.  What R can do that Excel can’t, is to go further to make great looking charts because of the customization that is possible.  I won’t claim it is easy, though.  The learning curve is [steep, but rewarding](http://r-statistics.co/Top50-Ggplot2-Visualizations-MasterList-R-Code.html).

![](http://r-statistics.co/screenshots/ggplot_masterlist_2.png)
[SOURCE](https://r-statistics.co/Top50-Ggplot2-Visualizations-MasterList-R-Code.html)

6. Scalability.  As your data sets become larger, R scales with them.  Excel becomes unwieldy.

7. Plug-in packages.  This is the single biggest reason to drop spreadsheets.  There are [hundreds of plug-in packages](https://cran.r-project.org/web/packages/available_packages_by_name.html) for R that extend its analytic power.  They can all be installed with a couple clicks.  Any new task I want to perform starts with asking if there is a package that will do it for me.  The answer is almost always yes.  Further, my own education in data science has advanced by leaps and bounds as I’ve learned to use these powerful new analytic tools.  I would go so far as to say this is a big career hack opportunity!  If you are producing highly sophisticated analyses you are going to get noticed compared to the person that is confined to the primitive capabilities of Excel.  

At the end of the day the tool that gets the job done is the right tool.  For me, R is a big step forward in efficiency, power and fun over Excel.

UPDATE June 2019: I originally wrote this note for an internal corporate blog in April 2017.  Then, I recommended to NOT to use R for real-time live dashboards.  I would amend that statement to say “it depends.”  The Shiny interactive web framework from RStudio makes interactive [dashboards look very good indeed](https://www.rstudio.com/products/shiny/shiny-user-showcase/).  Whether or not you can query live data APIs at the requisite frequency depends on the availability of a data feed, a package to grab it or your ability to write your own. 
