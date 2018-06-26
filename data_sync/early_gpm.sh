#####
args    = commandArgs(trailingOnly=T)
DOI     = as.Date(args[1])
yyyymm  = format(DOI,'%Y%m')
#####

prot = 'ftp://'
svr = 'jsimpson.pps.eosdis.nasa.gov'
pth = '/NRTPUB/imerg/early/'

OUTPUTDIR = paste0('/g/data/fj4/SatellitePrecip/GPM/global/early/',yyyymm)
if (!dir.exists(OUTPUTDIR)) {
        dir.create(OUTPUTDIR)
}

setwd(OUTPUTDIR)
commandText = paste0('/usr/bin/wget -nv -nd -N -o Download.log ',
                     prot,'luigi.j.renzullo%40gmail.com:luigi.j.renzullo%40gmail.com',
                     '@', svr, pth, yyyymm,'/3B*')

system(commandText)
