spawn,'ls U*/*txt > slist'

q=[0.47,1.12,1.77,2.50,3.19,3.90,4.58,$
   0.75,1.49,2.37,$
   0.10]

dgr=[0.01,0.010,0.0101,0.0102,0.0102,0.0103,0.0104,$
     0.00343,0.00344,0.00359,$
     0.00206]  ;ratio of dust mass to mass of H nucleons.

readcol,'slist',sname,f='A'
nline=file_lines(sname)
good=where(nline GT 100,nf)
sname=sname[good]

k=strip_path(sname)
k=repstr(k,'smc','smc_0')
;j=strpos(sname,'/')
;k=strarr(nf)
;for i=0,nf-1 do k[i]=strmid(sname[i],j[i]+2)
l=strarr(nf,4)
for i=0,nf-1 do l[i,*]=strsplit(k[i],'_',/extract)
l[*,3]=strmid(l[*,3],0,2)
l[*,0]=repstr(l[*,0],'U','')

umin=float(strcompress(l[*,0],/remove_all))
umax=float(l[*,1])
grain=strcompress(l[*,2],/remove_all)
qpah=fix(l[*,3])

qindex=qpah/10*(grain EQ 'MW3.1')+(qpah/5+7)*(grain EQ 'LMC2')+10*(grain EQ 'smc')



model={grain:'',umin:0.,umax:0.,qpah:0.,fnu_IRAC8:0.,fnu_IRS16:0.,$
       fnu_MIPS24:0.,fnu_MIPS70:0.,fnu_MIPS160:0.,$
       fnu_PACS70:0.,fnu_PACS100:0.,fnu_PACS160:0.,$
       fnu_SPIRE250:0.,fnu_SPIRE350:0.,fnu_spire500:0.,$
       slope:0.,wave:fltarr(1001),f_lambda:fltarr(1001)}

model=replicate(model,nf)
model.qpah=q[qindex]
model.umin=umin
model.umax=umax
model.grain=grain

m_p=1.67E-24 ;proton mass in grams
m_sun=1.988E33 ;solar mass in grams
lightspeed=2.998E18 ;in AA/s
G=1        ;;;;;4.*!PI*1E-23 ;convert from Jy/sr/H atom to erg/s/Hz/H atom  ;why is there an extra cm^2 factor?
K= alog10(m_sun)-alog10(m_p)-alog10(dgr[qindex])  ;convert from per/H atom to per/M_sun of dust.  grain dependent.
factor=alog10(G)-alog10(4.*!PI)-2.*alog10(!pc2cm*10) ;convert from erg/s to erg/s/cm^2 at 10pc

;stop

for i=0,nf-1 do begin
  nn=43
  djs_readcol,sname[i],w,nudpdnu,j_nu,skipline=nn+11,/silent

  if n_elements(w) LT 1E3 then begin
     nn=23 
     djs_readcol,sname[i],w,nudpdnu,j_nu,skipline=nn+11,/silent
  endif 

  djs_readcol,sname[i],band,pow,flux,numline=nn,skipline=11,/silent     

  model[i].fnu_irac8=flux[17]
  model[i].fnu_irs16=flux[21]
  model[i].fnu_mips24=flux[18]
  model[i].fnu_mips70=flux[19]
  model[i].fnu_mips160=flux[20]

  if nn GT 24 then begin
     model[i].fnu_pacs70=flux[23]
     model[i].fnu_pacs70=flux[23]
     model[i].fnu_pacs100=flux[24]
     model[i].fnu_pacs160=flux[25]
     model[i].fnu_spire250=flux[26]
     model[i].fnu_spire350=flux[27]
     model[i].fnu_spire500=flux[28]
  endif

;  print,sname[i],n_elements(w),w[987]


  w=w*1E4                                    ;AA
  nu=!lightspeed/w
  flambda=alog10(nudpdnu/w)+K[i]+factor ;erg/s/AA/cm^2/M_sun of dust at 10pc distance

  model[i].wave=reverse(w)
  model[i].f_lambda=reverse(10^flambda)

  gg=where(w GT 23E4 and w LT 30E4)
  gg=gg[sort(w[gg])]
  nu=lightspeed/w[gg]
  f_nu=f[gg]*1E31
  y=alog10(f_nu)
  x=alog10(nu)
  pp=linfit(x,y)
  model[i].slope=pp[1]
endfor


mwrfits,model,'DL07.fits',/create

end
