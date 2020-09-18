from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import las 
import base64
from sklearn import preprocessing
from sklearn.cluster import KMeans, MeanShift


app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["data"] = "./info"


@app.route('/', methods=['POST'])
def main():
    return render_template('home.html')

@app.route('/datos', methods=['GET','POST'])
def columnas():
    if request.method == 'POST':
        da = request.files['file']
        filename = secure_filename(da.filename)
        da.save(os.path.join(app.config["data"], filename))
        df = pd.read_csv('./info/{}'.format(filename))
        columnas = { 
            'columnas': df.columns.tolist(),
            'filename': filename
        }

        
        return render_template('datos.html', columnas=columnas)

@app.route('/grafica', methods=['POST'])
def grafica():    
    if request.method == 'POST':

        #LEER DATOS        
        filename = request.form['filename']

        try:
            salinidad = float(request.form['salinidad'])
        except:
            salinidad = 100
        try: 
            pma = float(request.form['dma'])
        except:
            pma=2.45
        try:
            pf = float(request.form['dfl'])
        except:
            pf=1.7
        try:        
            lma = float(request.form['lma'])
        except:
            lma=120
        try: 
            lfl = float(request.form['lfl'])
        except:
            lfl = 80
        try:
            ftort = float(request.form['ftort'])
        except:
            ftort = 1
        try:
            expc = float(request.form['expc'])
        except:
            expc = 2
        try:
            exps = float(request.form['exps'])
        except:
            exps = 2
        try:
            vgr = float(request.form['vgr'])
        except:
            vgr=1
        try:            
            vsp = float(request.form['vsp'])
        except:
            vsp=1
        try:
            vden = float(request.form['vden'])
        except:
            vden=0
        try:            
            vnphi = float(request.form['vnphi'])
        except:
            vnphi = 0
        try:        
            vrp = float(request.form['vrp'])
        except:
            vrp = 1
        try:
            vrm = float(request.form['vrm'])
        except:
            vrm = 1
        try:
            vrs = float(request.form['vrs'])
        except:
            vrs=1
        try:            
            nclust = int(request.form['nclust'])
        except:
            nclust = -1
        try:
            tipoVsh = request.form['tipo']        
        except:
            tipoVsh = 'lineal'

        #Buscar curvas 
        def buscarCurva(tipo, dicNombres, dh, unidad=False):    
            for nombre in dicNombres[tipo]["nombre"]:
                i=0
                for curva in dh.Nombre.values:
                    i+=1
                    #print (nombre, curva)
                    if (nombre in curva):
                        #print('Se encontró un valor por nombre')
                        return i-1    
            for desc in dicNombres[tipo]["desc"]:
                i=0
                for curva in dh.Descripcion.values:  
                    #print(desc, curva)
                    if (desc in curva):
                        #print('Se encontró un valor por descripción')
                        return i-1            
            if (unidad):
                for unidadad in dicNombres[tipo]["unidad"]:
                    i=0
                    for curva in dh.Unidades.values:  
                        i+=1
                        #print(desc, curva)
                        if (unidad in curva):
                            #print('Se encontró un valor por descripción')
                            return i-1
        #Cálculos
        #Temperatura
        def calcTempIntervalo(bht, ts, pm, cprof):
            ti= (((bht-ts)/pm)*cprof) + ts
            return ti
        #RESISTIVIDAD

        def calcRi(resistividad,curvaTemp,ts):
            #resistividad puede ser rmf, rmc, rm
            #6.77 es en Farenheit si TS está en grados debe cambiarse por 21.5
            Rint=resistividad*((ts+6.77)/(curvaTemp+6.77))
            return Rint

        def calcRmfEq(Rmfi, tf):
            temp1=Rmfi*(10**((0.426)/(np.log((tf)/(50.8)))))
            temp2=(1)/(np.log((tf)/(19.9)))
            temp3=0.131*(10**((temp2)-2))
            Rmfe=(temp1-temp3)/(1+(0.5*Rmfi))
            return Rmfe

        def calcRwEq(Rmfe, SSP):
            #Está en Farenheit
            #!!!!!!!!!!!!!!!!!!!!checar antes de entregar!!!!!!!!!!!!!!!!!!!!
            K=65+0.24
            Rwe = 10 ** ((K*np.log(Rmfe) + SSP)/(K))
            return Rwe

        def calcRw(Rwe, bht):
            temp1=Rwe + (0.131 * 10 ** ((1/(np.log(bht/19.9)))-2))
            temp2=-0.5*Rwe + (10**((0.0426)/(np.log(bht/50.8))))
            Rw=temp1/temp2
            return Rw

        def calcRxo(curvaProf, curvaSom):
            arr = abs(curvaProf-curvaSom)
            result = np.where(arr == np.amax(arr))
            return curvaSom[result[0][0]]

        #VOLUMEN DE ARCILLA
        def calcVArcilla(GR, metodo):
            IGR=(GR-min(GR))/(max(GR)-min(GR))
            if metodo == 'lineal':
                return IGR
            elif(metodo == 'larinovj'):
                Vsh = 0.083*((2 ** (3.71*IGR))-1)
                return Vsh
            elif(metodo == 'clavier'):
                Vsh = 1.7 * math.sqrt((3.38)*((IGR +0.7)**2))
                return Vsh
            elif (metodo=='larinovv'):
                Vsh = 0.33*((2 ** (2*IGR))-1)
                return Vsh

        #CORRECCIÓN DE POROSIDAD
        def calcCurvaPorDen(pma, pf, RHOB):    
            #Curva de porosidad densidad
            pord= (pma - RHOB)/(pma - pf)
            #pord>1=1
            return pord

        #Curva de porosidad total
        def calcPorTot(pord, NPHI):
            port = (pord - NPHI)/2
            return port

        #Curva de porosidad primaria o de matriz
        def calcPorP(lma, lfl, DT):
            porp = (DT - lma)/(lfl - lma)
            return porp

        #Curva de porosidad efectiva
        def calcPorEfec(port, Vsh):
            return (port*(1-Vsh))

        #Saturaciones
        def calcSw(a,m,n,Rw,Rt,por):
            temp1=Rt*((por)**m)
            Sw = ((a*Rw)/(temp1))**(1/n)
            return Sw

        def calcSxo(a,m,n,Rxo,Rmf,por):
            temp1=Rxo*((por)**m)
            Sxo = ((a*Rmf)/(temp1))**(1/n)
            return Sxo

        #Separación de capas
        def evaluarScore(score, porc):
            temp=100
            for i in range (len(score)):
                temp = abs((abs(score[i]) - abs(score[i+1]))*(100/score[i]))
                print (temp)
                if (temp<porc):
                    return i+1
                if (i==(len(score)-1)):
                    return i+1

        #Clasificación litológica
        def crearCurvas():
            pArena = np.array([[-1.5,2.66],[-1,2.64],[0.5, 2.605],[2,2.57],[5,2.51],[8,2.46],[10,2.43],[20,2.26],[25, 2.18],[31,2.08],[36,2.0],[40.5,1.92]])
            pCaliza = np.array([[3, 2.66],[36,2.1]])
            pDolomita = np.array([[4, 2.86],[8.5,2.82],[13, 2.76],[18,2.68],[23,2.59],[27,2.5],[32,2.4],[36,2.3],[38, 2.26],[41,2.18],[44,2.1]])
            pLutita = np.array([[30, 2.8],[40, 2.6],[41, 2.5]])
            
            arena=[]
            caliza = []
            dolomita = []
            lutita=[]

            za = np.polyfit(pArena[:,0], pArena[:,1], 5)
            pa = np.poly1d(za) 

            zc = np.polyfit(pCaliza[:,0], pCaliza[:,1], 1)
            pc = np.poly1d(zc)

            zd = np.polyfit(pDolomita[:,0], pDolomita[:,1], 5)
            pd = np.poly1d(zd)
            
            zl = np.polyfit(pLutita[:,0], pLutita[:,1], 1)
            pl = np.poly1d(zl)

            for i in range(45):
                arena.append([0.93*i-1.5, pa(0.93*i -1.5)])
                caliza.append([1*i, pc(1*i)])
                dolomita.append([0.94*i+2.5, pd(0.94*i +2.5)])
            for i in range(20):
                lutita.append([0.7*i+30, pc(1*i) + 0.1])

            arena=np.array(arena)
            caliza=np.array(caliza)
            dolomita=np.array(dolomita)
            lutita = np.array(lutita)
            
            return (arena, caliza, dolomita, lutita)

        def e_dist(a, b, metric='euclidean'):
            a = np.asarray(a)
            b = np.atleast_2d(b)
            a_dim = a.ndim
            b_dim = b.ndim
            if a_dim == 1:
                a = a.reshape(1, 1, a.shape[0])
            if a_dim >= 2:
                a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
            if b_dim > 2:
                b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
            diff = a - b
            dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
            if metric[:1] == 'e':
                dist_arr = np.sqrt(dist_arr)
            dist_arr = np.squeeze(dist_arr)
            return dist_arr

        def clasificarLito(arena, caliza, dolomita, lutita, dato1, dato2):
            minimo=[]
            
            tarena=np.copy(arena)
            tcaliza=np.copy(caliza)
            tdolomita=np.copy(dolomita)
            tlutita=np.copy(lutita)
            
            tarena[:,0]=arena[:,0]/10
            tcaliza[:,0]=caliza[:,0]/10
            tdolomita[:,0]=dolomita[:,0]/10
            tlutita[0][0]=lutita[0][0]/10
            dato1=dato1/10
            
            minArena=(e_dist([dato1,dato2],tarena))
            minCaliza=(e_dist([dato1,dato2],tcaliza))
            minDolomita=(e_dist([dato1,dato2],tdolomita))
            minLutita=(e_dist([dato1,dato2],tlutita))

            minimo.append(min(minArena))
            minimo.append(min(minCaliza))
            minimo.append(min(minDolomita))
            minimo.append(min(minLutita))    
            
            #return minimo
            
            if np.argmin(minimo)==0:
                mini = np.argmin(minArena)
                return ("Arena",mini)
            elif np.argmin(minimo)==1:
                mini = np.argmin(minCaliza)
                return ("Caliza",mini)
            elif np.argmin(minimo)==2:
                mini = np.argmin(minDolomita)
                return ("Dolomita",mini)
            elif np.argmin(minimo)==3:
                mini = 0
                return ("Lutita",mini)
            
        def dibujarRegistros(df):
            fig, ax = plt.subplots(nrows=1, ncols=len(df.columns.values), figsize=(20,10), sharey=True)
            fig.suptitle("Registros geofísicos de pozos", fontsize=22)
            fig.subplots_adjust(top=0.75,wspace=0.2)
            i=0
            
            if 'Clasif' in df.columns:
                for registro in (df.columns.values[:-1]):        
                    color='black'
                    ax10=ax[i].twiny()
                    #ax10.set_xlim(min(df[registro]),max(df[registro]))
                    #ax10.spines['top'].set_position(('outward',0))
                    ax10.plot(df[registro],df.index.values, color=color)
                    ax10.set_xlabel(registro+' ['']',color=color)    
                    #ax10.tick_params(axis='x', colors=color)
                    ax10.invert_yaxis()
                    ax10.grid(True)
                    i+=1        
                ax10=ax[len(df.columns)-1].twiny()
                a = df.Clasif.values
                data=df.Clasif.values.reshape(len(df.Clasif.values), 1)
                cmap = plt.get_cmap('Dark2', np.max(data)-np.min(data)+1)        
                mat = ax10.matshow(np.repeat(data, 300, 1),cmap=cmap,vmin = np.min(data)-.5, vmax = np.max(data)+.5)        
                cax = plt.colorbar(mat, ticks=np.arange(np.min(data),np.max(data)+1))            
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                ax10.set_xlabel(registro+' ['']',color=color)    
                #ax.figure.set_size_inches(300, 500)
                #ax10.tick_params(axis='x', colors=color)
                ax10.invert_yaxis()
                ax10.grid(True)
                i+=1
                
            else:
                for registro in (df.columns.values):        
                    color='black'
                    ax10=ax[i].twiny()
                    #ax10.set_xlim(min(df[registro]),max(df[registro]))
                    #ax10.spines['top'].set_position(('outward',0))
                    ax10.plot(df[registro],df.index.values, color=color)
                    ax10.set_xlabel(registro+' ['']',color=color)    
                    #ax10.tick_params(axis='x', colors=color)
                    ax10.invert_yaxis()
                    ax10.grid(True)
                    i+=1


        #LEER ARCHIVO        
        log = las.LASReader('./info/{}'.format(filename))
        dat=pd.DataFrame(data=log.data, columns=log.curves.names)
        dat=dat.replace(-999.00000,0.0)

        #Se guarda el header en un diccionario
        dic=log.curves.items

        #Se crean arreglos de nombre, unidades y descripción
        ldescr = []
        lunidades = []
        lnombres=log.curves.names 

        #Se guardan los datos dentro del arreglo
        for i in range (len(lnombres)):
            ldescr.append(dic[log.curves.names[i]].descr.upper())
            lunidades.append(dic[log.curves.names[i]].units.upper())   

        data = {"Nombre": lnombres, "Descripcion": ldescr, "Unidades": lunidades}
        dh = pd.DataFrame(data)  
        
        dicNombres = {
            "profundidad": {
                'nombre' : ["DEPT", "DEPTH"],
                'desc' : ["DEPT", "DEPTH"],
                #FALSE
                'unidad' : ["FT", "M"]
            },
            "caliper": {
                'nombre' : ["CALIPER", "CALI","CAL","DAC","MSC","CL","TAC","MCT","EMS","CCT","XYT","CCN","DNSC","DSNCM"],
                'desc' : ["CALIPER", "CALI"],
                'unidad' : ["IN"]
            },
            "sp": {
                'nombre' : ["SP"],
                'desc' : ["SP", "SPONTANEUS", "POTENCIAL"],
                'unidad' : ["MV", "V"]
            },
            "gr": {
                'nombre' : ["GR","MCG","MGS","NGS","NGT","IPL","GRT","DGR","DG","SL","HDS1","RGD","CWRD","SGR"],
                'desc' : ["GR", "GAMMA", "RAY"],
                'unidad' : ["GAPI", "API"]
            },
            "rhob": {
                'nombre' : ["RHOB", "APLS", "ZDL", "CDL", "SPeD", "SDL","PDS", "MPD","IPL","CDT","LDT","ORD","MDL","DNSC","ASLD"],
                'desc' : ["DENSITY","RHOB", "RHO"],
                'unidad' : ["G/C3"]
            },
            "nphi": {
                'nombre' : ["NPHI", "NPH", "CN","DSN","DSEN","MDN","IPL","CNT","CCN","MNP","DNSC","CTN"],
                'desc' : ["NEUTRON","NEUT"],
                #FALSE
                'unidad' : ["V/V"]
            },
            "rsom": {
                'nombre' : ["LL3","SGRD","SFL","SLL","LLS","RLLS"],
                'desc' : ["SHALL"],
                #FALSE
                'unidad' : ["OHMMxxx"]
            },
            "rmed": {
                'nombre' : ["R60O","ILM","RILM"],
                'desc' : ["MEDR","MED"],
                #FALSE
                'unidad' : ["OHMMxxxx"]
            },
            "rprof": {
                'nombre' : ["R85O","ILD","RILD","DLL","LLD","RLLD"],
                'desc' : ["DEEPR","DEEP"],
                #FALSE
                'unidad' : ["OHMMxxxx"]
            },
            "dt": {
                'nombre' : ["DT","APX","XMAC","DAL","AC","BCS","DAR","FWS","XACT","CSS","LCS","MSS","UGD","DSI","CST","LST","DNSC","SONIC","BAT"],
                'desc' : ["DT","SONIC"],    
                'unidad' : ["US/F"]
            },  
        }

        n1 = ['DEPTH','CALIPER', 'GR', 'SP', 'RHOB', 'NPHI', 'RCERC','RMED', 'RPROF', 'DT']
        n2 = ['profundidad','caliper','gr','sp','rhob','nphi', 'rsom', 'rmed', 'rprof', 'dt']
        noSePuede = []
        data={}

        #Se rellena el vector "No se pudo encontrar"
        for i in range(len(n1)):
            try:
                data[n1[i]] = dat[log.curves.names[buscarCurva(n2[i], dicNombres, dh)]]
                #print ('Se encontró la curva '+n2[i])
            except:
                noSePuede.append(n1[i])
                #print ('No se han encontrado '+n2[i])

        calcularSat=True
        calcularXPlot=True
        calcularRw = True
        calcularVsh = True
        calcularRxo = True
        calcularPort = True
        calcularPore = True
        calcularSw = True
        calcularSxo = True

        #Se determina qué cálculos se pueden realizar con base en las que se pudieron encontrar
        if ('DEPTH' in noSePuede or 'SP' in noSePuede):
            calcularRw = False
            calcularSat = False
            calcularSw = False
        if ('RPROF' in noSePuede):
            calcularSat = False
            calcularSw = False
        if ('RHOB' in noSePuede or 'NPHI' in noSePuede):
            calcularXPlot = False
            calcularPort = False
            calcularSw = False
            calcularSxo = False
        if ('GR' in noSePuede):
            calcularVsh = False
            calcularPore = False
        if ('RCERC' in noSePuede):
            calcularRxo = False
            calcularSxo = False
        
        df = pd.DataFrame(data=data) 

        #Datos del archivo .LAS
        try:
            rmf=float(log.parameters.RMF.data)
        except: 
            rmf=0.95
        try:
            rmc=float(log.parameters.RMC.data)
        except: 
            rmc=1.55
        try:
            rm=float(log.parameters.RM.data)
        except: 
            rm=1.13
        try:
            bht=float(log.parameters.BHT.data)
        except: 
            bht=3000
        try:
            ts=float(log.parameters.MST.data)
        except: 
            ts=3000
        try:     
            pm=max(df.DEPTH)
        except:
            pm=1000

        #Aplicación de correcciones
        if ('GR' in df.columns):
            df['GR']=df['GR']*vgr
        if ('SP' in df.columns):
            df['SP']=df['SP']*vsp
        if ('RHOB' in df.columns):
            df['RHOB']=df['RHOB']+vden
        if ('NPHI' in df.columns):
            df['NPHI']=df['NPHI']+vnphi
        if ('RCERC' in df.columns):
            df['RCERC']=df['RCERC']*vrs
        if ('RMED' in df.columns):
            df['RMED']=df['RMED']*vrm
        if ('RPROF' in df.columns):
            df['RPROF']=df['RPROF']*vrp
        df1 = df[df.columns[1:]].copy()
        calcularXPlot=True
        if (calcularVsh):
            df['VSH']=calcVArcilla(df.GR, tipoVsh)
        if (calcularPort):
            pord=calcCurvaPorDen(pma, pf, df.RHOB)
            df['PTOT']=calcPorTot(pord, df.NPHI)
        if (calcularPore):
            df['PEfec']=calcPorEfec(df.PTOT, df.VSH)           
        if (calcularRw):
            df['TEMP']=calcTempIntervalo(bht, ts, pm, df.DEPTH)
            curvaRmf = calcRi(rmf, df.TEMP, ts)
            tf = df.TEMP.values[df.SP.idxmin()]
            rmEq = calcRmfEq(curvaRmf, tf)
            rwEq = calcRwEq(rmEq, df.SP)
            curvaRw=calcRw(rwEq, bht)
            Rw=curvaRw[df.SP.idxmin()]
        if (calcularRxo):
            Rxo = calcRxo(df['RPROF'], df['RCERC'])           
        if (calcularSw):
            df['Sw']=calcSw(ftort,expc,exps,Rw, df['RPROF'],df['PTOT'])
        if (calcularSxo):
            df['Sxo']=calcSxo(a,m,n,Rxo,rmf,df['PTOT'])


        #Se crea un dataframe y se realiza un preprocesamiento para su clasificación
        x = df1.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df1nor = pd.DataFrame(x_scaled)
        df1nor.columns = df1.columns

        # #Se obtiene un arreglo de datos del dataframe de datos normalizados
        X=np.array(df1nor)

        # Se obtiene una predicción del modelo por cada muestra
        if (nclust == -1):
            #Se ajusta el número de clusters con la información del dataframe    
            labels=MeanShift().fit_predict(X)    
        else:
            kmeans = KMeans(n_clusters=nclust).fit(X)
            centroides = kmeans.cluster_centers_
            labels = kmeans.predict(X)

        #Se agrega la clasificación realizada al dataframe original
        df['Clasif'] = labels
        #Se cuenta el número de clusters 
        ncl=len(np.unique(labels))

        #Creación de puntos de clasificación
        ptos = np.zeros(shape=(ncl,2))

        for i in range (ncl):
            temp=df[df.Clasif == i]['RHOB'].mean()
            temp2=df[df.Clasif == i]['NPHI'].mean()
            ptos[i]=[temp, temp2]    
            
        if (max(abs(ptos[:,1]))<1):
            ptos[:,1]=ptos[:,1]*100

        #Se crean las curvas de cada litología
        [arena, caliza, dolomita, lutita]=crearCurvas()

        lit=[]
        por=[]
        for i in range(len(ptos)):
            [temp1, temp2]=clasificarLito(arena, caliza, dolomita, lutita, ptos[i, 1], ptos[i,0])
            lit.append(temp1)
            por.append(temp2)
        

        #GRAFICAR
        plt.clf()  

        #Imagen 1  
        img = io.BytesIO()

        #Gráfica
        i=0

        #Numero de gráficos
        if ('CALIPER' in df.columns or 'SP' in df.columns):    
            i+=1
        if ('GR' in df.columns):
            i+=1
        if ('RCERC' in df.columns or 'RMED' in df.columns or 'RPROF' in df.columns):    
            i+=1
        if ('NPHI' in df.columns or 'RHOB' in df.columns):    
            i+=1
        if ('Clasif' in df.columns):
            i+=1
        if ('DT' in df.columns):  
            i+=1
        if ('PTOT' in df.columns or 'PEfec' in df.columns):  
            i+=1
        if ('Sw' in df.columns or 'Sxo' in df.columns):  
            i+=1

        fig, ax = plt.subplots(nrows=1, ncols=i, figsize=(20,10), sharey=True)
        fig.suptitle("Registros geofísicos de pozos", fontsize=22)
        fig.subplots_adjust(top=0.75,wspace=0.2)

        for axes in ax:
                axes.set_ylim (log.start,log.stop)
                axes.invert_yaxis()
                axes.yaxis.grid(True)
                axes.get_xaxis().set_visible(False)
        i=-1

        #CARRIL 1
        #CALI
        if ('CALIPER' in df.columns or 'SP' in df.columns):    
            i+=1

        if ('CALIPER' in df.columns):    
            color='black'
            ax1=ax[i].twiny()
            ax1.set_xlim(min(df.CALIPER),max(df.CALIPER))
            ax1.spines['top'].set_position(('outward',0))
            ax1.plot(df.CALIPER, df.DEPTH, '--', color=color)
            ax1.set_xlabel('CALIPER',color=color)    
            ax1.tick_params(axis='x', colors=color)
            ax1.grid(True)
        #SP
        if ('SP' in df.columns):
            color='blue'
            ax2=ax[i].twiny()
            ax2.set_xlim(min(df.SP),max(df.SP))
            ax2.spines['top'].set_position(('outward',40))
            ax2.plot(df.SP, df.DEPTH, color=color)
            ax2.set_xlabel('SP',color=color)    
            ax2.tick_params(axis='x', colors=color)
            ax2.grid(True)


        #CARRIL 2
        #GR
        if ('GR' in df.columns):
            i+=1
            color='green'
            ax3=ax[i].twiny()
            ax3.set_xlim(min(df.GR),max(df.GR))
            ax3.spines['top'].set_position(('outward',0))
            ax3.plot(df.GR, df.DEPTH, color=color)
            ax3.set_xlabel('GR',color=color)    
            ax3.tick_params(axis='x', colors=color)
            ax3.grid(True)


        # #CARRIL 3
        if ('RCERC' in df.columns or 'RMED' in df.columns or 'RPROF' in df.columns):    
            i+=1
        #RPROFUNDA
        if ('RPROF' in df.columns):    
            color='red'
            ax4=ax[i].twiny()
            ax4.set_xlim(0.1,10000)
            ax4.set_xscale('log')
            ax4.grid(True)
            ax4.spines['top'].set_position(('outward',0))
            ax4.set_xlabel('RPROF',color=color)    
            ax4.plot(df.RPROF ,df.DEPTH, color=color)
            ax4.tick_params(axis='x', colors=color)  

        #RMEDIA
        if ('RMED' in df.columns):    
            color='orange'
            ax5=ax[i].twiny()
            ax5.set_xlim(0.1,10000)
            ax5.set_xscale('log')
            ax5.grid(True)
            ax5.spines['top'].set_position(('outward',40))
            ax5.set_xlabel('RMED',color=color)    
            ax5.plot(df.RMED ,df.DEPTH, color=color)
            ax5.tick_params(axis='x', colors=color)  

        #RSOMERA
        if ('RCERC' in df.columns):    
            color='yellow'
            ax6=ax[i].twiny()
            ax6.set_xlim(0.1,10000)
            ax6.set_xscale('log')
            ax6.grid(True)
            ax6.spines['top'].set_position(('outward',80))
            ax6.set_xlabel('RSOM',color=color)    
            ax6.plot(df.RCERC ,df.DEPTH, color=color)
            ax6.tick_params(axis='x', colors=color)  


        #CARRIL 4
        if ('NPHI' in df.columns or 'RHOB' in df.columns):    
            i+=1
        #NPHI
        if ('NPHI' in df.columns):   
            color='blue'
            ax7=ax[i].twiny()
            ax7.set_xlim(0.45,-0.15)
            ax7.invert_xaxis()
            ax7.plot(df.NPHI, df.DEPTH, color=color) 
            ax7.spines['top'].set_position(('outward',0))
            ax7.set_xlabel('NPHI',color=color)   
            ax7.tick_params(axis='x', colors=color)

        #RHOB
        if ('RHOB' in df.columns):   
            color2='red'
            ax8=ax[i].twiny()
            ax8.set_xlim(min(df.RHOB),max(df.RHOB))
            ax8.plot(df.RHOB, df.DEPTH ,label='RHOB', color=color2) 
            ax8.spines['top'].set_position(('outward',40))
            ax8.set_xlabel('RHOB',color=color2)   
            ax8.tick_params(axis='x', colors=color2)
            
            
        #CARRIL 5
        #DT
        if ('DT' in df.columns):  
            i+=1
            color='purple'
            ax9=ax[i].twiny()
            ax9.set_xlim(min(df.DT),max(df.DT))
            ax9.invert_xaxis()
            ax9.plot(df.DT, df.DEPTH, color=color) 
            ax9.spines['top'].set_position(('outward',0))
            ax9.set_xlabel('DT',color=color)   
            ax9.tick_params(axis='x', colors=color)    
            
            
        #CARRIL 6
        #Porosidad total
        if ('PTOT' in df.columns or 'PEfec' in df.columns):  
            i+=1
        if ('PTOT' in df.columns):  
            color='red'
            ax10=ax[i].twiny()
            ax10.set_xlim(1,0)
            ax10.spines['top'].set_position(('outward',0))
            ax10.plot(df.PTOT, df.DEPTH, color=color)
            ax10.fill_betweenx(df.DEPTH,0,df.PTOT,color='lightcoral')
            ax10.set_xlabel('P.Tot',color=color)   
            ax10.tick_params(axis='x', colors=color)
            ax10.grid(True)
        if ('PTOT' in df.columns):  
            color='blue'
            ax11=ax[i].twiny()
            ax11.set_xlim(1,0)
            ax11.spines['top'].set_position(('outward',40))
            ax11.plot(df.PEfec, df.DEPTH, color=color)
            ax11.fill_betweenx(df.DEPTH,0,df.PEfec,color='lightblue')
            ax11.set_xlabel('P.Efec',color=color)   
            ax11.tick_params(axis='x', colors=color)
            ax11.grid(True)


        #CARRIL 7
        if ('Sw' in df.columns or 'Sxo' in df.columns):  
            i+=1
        if ('Sw' in df.columns):
            color='blue'
            ax11=ax[i].twiny()
            ax11.set_xlim(min(df.Sw),max(df.Sw))
            ax11.spines['top'].set_position(('outward',0))
            ax11.plot(df.Sw, df.DEPTH, color=color)
            ax11.set_xlabel('Sw',color=color)   
            ax11.tick_params(axis='x', colors=color)
            ax11.grid(True)
        if ('Sxo' in df.columns):
            color='lightgreen'
            ax11=ax[i].twiny()
            ax11.set_xlim(min(df.Sxo),max(df.Sxo))
            ax11.spines['top'].set_position(('outward',40))
            ax11.plot(df.Sxo, df.DEPTH, color=color)
            ax11.set_xlabel('Sxo',color=color)   
            ax11.tick_params(axis='x', colors=color)
            ax11.grid(True)
            


        # CARRIL 8
        if ('Clasif' in df.columns):
            i+=1
            X=np.arange(0,1,0.1)
            Y=df.DEPTH.values
            Z=df.Clasif.values
            Z=Z.reshape(len(Z),1)
            Z2=np.repeat(Z, 10, 1)

            ax20=ax[i]
            cmap = plt.get_cmap('Dark2', np.max(Z)-np.min(Z)+1)  
            c = ax20.pcolor(X,Y,Z2, cmap=cmap,vmin=np.min(Z)-.5, vmax=np.max(Z)+.5)
            cbar = fig.colorbar(c, ax=ax20, ticks=np.arange(np.min(Z),np.max(Z)+1))
            cbar.ax.set_yticklabels(lit) 
            #fig.colorbar(c, ax=ax20, ticks=['Arena','Caliza','Lutita'])
            
        #     cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
        #     cbar.ax.set_yticklabels(['< -1', '0', '> 1']) 
       

        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        # session['data']=plot_url


        #Imagen 2

        img2 = io.BytesIO()
        fig, ax = plt.subplots(figsize=(15,7),)

        #plt.scatter (pArena[:,0], pArena[:,1])
        ax.plot(arena[:,0], arena[:,1],'--', label='Arenisca')
        ax.plot(caliza[:,0], caliza[:,1],'--', label='Caliza')
        ax.plot(dolomita[:,0], dolomita[:,1],'--', label='Dolomita')
        ax.scatter(ptos[:,1], ptos[:,0])
        ax.scatter(lutita[9,0], lutita[9,1], label='Lutita')
        # plt.scatter(lutita[0][0], lutita[0][1], label='Lutita')
        ax.set_ylabel('Densidad [g/cm3]')
        ax.set_xlabel('Porosidad de neutrón')
        ax.invert_yaxis()
        ax.legend()
        ax.grid()

        plt.savefig(img2, format='png')
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()
        
        return render_template('grafica.html', imagen={ 'imagen': plot_url, 'imagen2': plot_url2 })

if __name__ == "__main__":    
    app.run(port = 80, debug = True)

