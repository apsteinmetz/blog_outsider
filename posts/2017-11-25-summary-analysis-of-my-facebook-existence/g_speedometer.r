#google bar gauge
#http://www.gastonsanchez.com/visually-enforced/how-to/2013/01/10/Gauge-Chart-in-R/
# using base R

#function to create a circle
circle <- function(center=c(0,0), radius=1, npoints=100)
{
  r = radius
  tt = seq(0, 2*pi, length=npoints)
  xx = center[1] + r * cos(tt)
  yy = center[1] + r * sin(tt)
  return(data.frame(x = xx, y = yy))
}

# function to get slices
slice2xy <- function(t, rad) 
{
  t2p = -1 * t * pi + 10*pi/8
  list(x = rad * cos(t2p), y = rad * sin(t2p))
}

# function to get major and minor tick marks
ticks <- function(center = c(0, 0), from = 0, to = 2*pi, 
                  radius = 0.9, npoints = 5)
{
  r = radius
  tt = seq(from, to, length = npoints)
  xx = center[1] + r * cos(tt)
  yy = center[1] + r * sin(tt)
  return(data.frame(x = xx, y = yy))
}

# external circle (this will be used for the black border)
border_cir = circle(c(0, 0), radius = 1, npoints = 100)

# gray border circle
external_cir = circle(c(0,0), radius = 0.97, npoints = 100)

# ---------------------------------------------------
get_color_slice_range<-function(from = 75,to = 90){
# get coordinates for a desired percentage range to use for a color band
  band_ini = (from / 100) * (12/8)
  band_fin = (to / 100) * (12/8)
  Slice = slice2xy(seq.int(band_ini, band_fin, length.out = 30), rad = 0.9)
  return(Slice)
}
# ---------------------------------------------------


# coordinates of major ticks (will be plotted as arrows)
major_ticks_out = ticks(c(0,0), from = 5*pi/4, to = -pi/4, radius = 0.9, 5)
major_ticks_in = ticks(c(0,0), from = 5*pi/4, to = -pi/4, radius = 0.75, 5)

# coordinates of minor ticks (will be plotted as arrows)
tix1_out = ticks(c(0, 0), from = 5*pi/4, to = 5*pi/4-3*pi/8, radius = 0.9, 6)
tix2_out = ticks(c(0, 0), from = 7*pi/8, to = 7*pi/8-3*pi/8, radius = 0.9, 6)
tix3_out = ticks(c(0, 0), from = 4*pi/8, to = 4*pi/8-3*pi/8, radius = 0.9, 6)
tix4_out = ticks(c(0, 0), from = pi/8, to = pi/8-3*pi/8, radius = 0.9, 6)
tix1_in = ticks(c(0, 0), from = 5*pi/4, to = 5*pi/4-3*pi/8, radius = 0.85, 6)
tix2_in = ticks(c(0, 0), from = 7*pi/8, to = 7*pi/8-3*pi/8, radius = 0.85, 6)
tix3_in = ticks(c(0, 0), from = 4*pi/8, to = 4*pi/8-3*pi/8, radius = 0.85, 6)
tix4_in = ticks(c(0, 0), from = pi/8, to = pi/8-3*pi/8, radius = 0.85, 6)

# coordinates of min and max values (0, 100)
v0 = -1 * 0 * pi + 10*pi/8
z0x = 0.65 * cos(v0)
z0y = 0.65 * sin(v0)
v100 = -1 * 12/8 * pi + 10*pi/8
z100x = 0.65 * cos(v100)
z100y = 0.65 * sin(v100)



add_color_band<-function(color="white",start=0,stop=100){
  coord<-get_color_slice_range(start,stop)
  polygon(c(coord$x, 0),
          c(coord$y, 0),
          border= color,
          col=color,
          lty=NULL)
  
}

plot_speedometer<-function(label,
                           value,
                           bottom_label=value,
                           min_val,max_val){
#value is as a percent of 0-100. values are absolute  
  
  #rescale values to a percent of 100
  rel_val<-round((value-min_val)/(max_val-min_val)*100)
  # open plot
  plot(border_cir$x, border_cir$y, type="n", asp=1, axes=FALSE,
       xlim=c(-1.05,1.05), ylim=c(-1.05,1.05),
       xlab="", ylab="")
  # green slice
  add_color_band("green",50,100)
  # yellow slice
  add_color_band("#FF9900",10,50)
  # red slice
  add_color_band("#DC3912",0,10)
  # add gray border
  lines(external_cir$x, external_cir$y, col = "gray85", lwd = 20)
  # add external border
  lines(border_cir$x, border_cir$y, col = "gray20", lwd = 2)
  # add minor ticks
  arrows(x0 = tix1_out$x, y0 = tix1_out$y, x1 = tix1_in$x, y1 = tix1_in$y,
         length = 0, lwd = 2.5, col = "gray55")
  arrows(x0 = tix2_out$x, y0 = tix2_out$y, x1 = tix2_in$x, y1 = tix2_in$y,
         length = 0, lwd = 2.5, col = "gray55")
  arrows(x0 = tix3_out$x, y0 = tix3_out$y, x1 = tix3_in$x, y1 = tix3_in$y,
         length = 0, lwd = 2.5, col = "gray55")
  arrows(x0 = tix4_out$x, y0 = tix4_out$y, x1 = tix4_in$x, y1 = tix4_in$y,
         length = 0, lwd = 2.5, col = "gray55")
  # add major ticks
  arrows(x0 = major_ticks_out$x, y0 = major_ticks_out$y,
         x1 = major_ticks_in$x, y1 = major_ticks_in$y, length = 0, lwd = 4)
  # angle of needle pointing to the specified value
  val = (rel_val/100) * (12/8)
  v = -1 * val * pi + 10*pi/8
  # x-y coordinates of needle
  val_x = 0.7 * cos(v)
  val_y = 0.7 * sin(v)
  # add needle
  arrows(0, 0, val_x, val_y, col = "#2e9ef3", lwd = 7)
  # add central blue point
  points(0, 0, col = "#2e9ef3", pch = 19, cex = 5)
  # add label of variable at top of gauge
  text(0, 0.43, label, cex = 2.0)
  # add signed value at bottom of gauge
  text(0, -0.65, bottom_label, cex = 2.0)
  # add values 0 and 100
  text(z0x, z0y, labels = min_val, col = "gray50",cex=2.0)
  text(z100x, z100y, labels = max_val, col = "gray50",cex=2.0)
  
  
}

#plot_speedometer(label="Mood",value=0.15,min=-5,max=5)
