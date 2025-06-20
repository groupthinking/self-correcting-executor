import { motion } from 'framer-motion';

const ComponentManager = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="glass-card"
    >
      <h2>Component Manager</h2>
      <p>Manage protocols, agents, and connectors</p>
    </motion.div>
  );
};

export default ComponentManager;
