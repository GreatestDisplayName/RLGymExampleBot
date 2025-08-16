const {createApp} = Vue;
createApp({
  data(){ return {env:"CartPole-v1", algo:"DQN", max:1000, runId:null, chart:null, points:[]}; },
  mounted(){
    this.chart = new Chart(document.getElementById('chart'),{type:'line', data:{labels:[],datasets:[{label:'Reward',data:[]}]}});
  },
  methods:{
    async createRun(){
      const fd=new FormData();
      fd.append("env", this.env);
      fd.append("algo", this.algo);
      fd.append("max_iter", this.max);
      const res = await fetch("/create",{method:"POST",body:fd});
      const {run_id} = await res.json();
      this.runId = run_id;
      this.points = [];
      const ws = new WebSocket(`ws://localhost:8000/ws/${run_id}`);
      ws.onmessage = (ev) => {
        const data=JSON.parse(ev.data);
        if(data.type==='log'){
          this.chart.data.labels.push(Date.now());
          this.chart.data.datasets[0].data.push(data.reward);
          this.chart.update();
        }
        if(data.type==='done') ws.close();
      };
    }
  }
}).mount('#app');